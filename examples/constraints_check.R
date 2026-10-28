# R-side oracle for monotone constraints: gbm's `monotone` argument uses the
# same -1/0/+1 convention as our ModelTree. Ground truth for
# examples/constraints_check.py.
#   R_LIBS_USER=$HOME/R/library Rscript examples/constraints_check.R
#
# Domain fact: in healthcare_insurance_expenses, medical-cost literature says
# charges rise with age -> constraint +1 on age.

data_dir <- file.path("benchmarks", "elo_data")
train <- read.csv(file.path(data_dir, "ta_healthcare_train.csv"), check.names = FALSE)
test <- read.csv(file.path(data_dir, "ta_healthcare_test.csv"), check.names = FALSE)
ycol <- "y"
xs <- setdiff(names(train), ycol)
age_col <- which(xs == "age") - 1L  # gbm monotone indexes predictors 1..p -> R positions

mono <- rep(0L, length(xs))
mono[age_col + 1L] <- 1L

# Probe grid: age swept across its range, other columns at their medians.
ages <- seq(18, 64, length.out = 24)
med <- sapply(xs, function(n) median(train[[n]], na.rm = TRUE))
grid <- as.data.frame(matrix(rep(med, each = length(ages)), nrow = length(ages)))
names(grid) <- xs
grid[["age"]] <- ages

fit_pred <- function(mono_v) {
  m <- gbm::gbm(
    as.formula(paste(ycol, "~ .")), data = train,
    distribution = "gaussian", n.trees = 500, interaction.depth = 4,
    shrinkage = 0.05, bag.fraction = 0.5, train.fraction = 1.0,
    var.monotone = mono_v, verbose = FALSE
  )
  pred_grid <- predict(m, grid, n.trees = 500)
  pred_test <- predict(m, test[, xs], n.trees = 500)
  list(grid = as.numeric(pred_grid), test = as.numeric(pred_test))
}

u <- fit_pred(rep(0L, length(xs)))
c <- fit_pred(mono)

mono_ok <- function(p) all(diff(p) >= -1e-9)
rmse <- function(p, y) sqrt(mean((p - y)^2))

cat(sprintf("R gbm (oracle)   unconstrained monotone: %s   constrained: %s\n",
            mono_ok(u$grid), mono_ok(c$grid)))
cat(sprintf("R gbm (oracle)   rmse uncon %.3f -> constrained %.3f\n",
            rmse(u$test, test[[ycol]]), rmse(c$test, test[[ycol]])))
cat("R gbm (oracle)   constrained grid predictions:\n")
cat(paste(sprintf("%.2f", c$grid), collapse = " "), "\n")
