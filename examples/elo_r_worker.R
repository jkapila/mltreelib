# Fit one R (algorithm x dataset) as a secondary command-line process, run
# under the budget harness (Rscript --vanilla in its own process group).
# setTimeLimit stays as a cooperative floor; the hard kill is external.
#
#   Rscript --vanilla examples/elo_r_worker.R <dataset> <model>
#
# prints exactly one line: RESULT dataset,model,rmse,mae,fit_s

args <- commandArgs(trailingOnly = TRUE)
ds <- args[[1]]
model <- args[[2]]
data_dir <- file.path("benchmarks", "elo_data")

train <- read.csv(file.path(data_dir, paste0(ds, "_train.csv")),
                  check.names = FALSE)
test <- read.csv(file.path(data_dir, paste0(ds, "_test.csv")),
                 check.names = FALSE)
ycol <- "y"
xs <- setdiff(names(train), ycol)

setTimeLimit(elapsed = 170, transient = TRUE)  # cooperative floor (hard cap 180s external)

t0 <- proc.time()[["elapsed"]]
pred <- switch(model,
  rpart = {
    fit <- rpart::rpart(as.formula(paste(ycol, "~ .")), data = train)
    predict(fit, test)
  },
  cubist = {
    fit <- Cubist::cubist(x = train[, xs, drop = FALSE], y = train[[ycol]])
    predict(fit, test[, xs, drop = FALSE])
  },
  lmtree = {
    f <- as.formula(paste(ycol, "~", paste0("`", xs, "`", collapse = " + "), "|",
                          paste0("`", xs, "`", collapse = " + ")))
    fit <- partykit::lmtree(f, data = train)
    predict(fit, test)
  },
  glmtree = {
    f <- as.formula(paste(ycol, "~", paste0("`", xs, "`", collapse = " + "), "|",
                          paste0("`", xs, "`", collapse = " + ")))
    fit <- partykit::glmtree(f, data = train)
    predict(fit, test, type = "response")
  },
  ctree = {
    fit <- partykit::ctree(as.formula(paste(ycol, "~ .")), data = train)
    predict(fit, test)
  },
  grf_llf = {
    fit <- grf::ll_regression_forest(as.matrix(train[, xs, drop = FALSE]),
                                     train[[ycol]])
    predict(fit, as.matrix(test[, xs, drop = FALSE]))$predictions
  },
  grf_rf = {
    fit <- grf::regression_forest(as.matrix(train[, xs, drop = FALSE]),
                                  train[[ycol]])
    predict(fit, as.matrix(test[, xs, drop = FALSE]))$predictions
  },
  earth = {
    fit <- earth::earth(as.formula(paste(ycol, "~ .")), data = train)
    predict(fit, test)
  },
  stop("unknown model: ", model)
)
dt <- proc.time()[["elapsed"]] - t0

pred <- as.numeric(pred)
ok <- is.finite(pred)
if (!any(ok)) stop("no finite predictions")
rmse <- sqrt(mean((pred[ok] - test[[ycol]][ok])^2))
mae <- mean(abs(pred[ok] - test[[ycol]][ok]))
cat(sprintf("RESULT %s,%s,%.4f,%.4f,%.3f\n", ds, model, rmse, mae, dt))
