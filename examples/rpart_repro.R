# rpart ground truth for the mltreelib model-tree sanity check.
#
# Runs rpart LIVE on the canonical mtcars regression task with growth-only
# controls (cp = 0; pruning/cost-complexity is a later step in our plan), and
# prints exact node/split values for comparison with our tree. Also runs an NA
# variant (wt knocked out) — rpart routes NAs via surrogate splits, our tree
# via learned default directions + NA-indicator leaf terms.
#
#   Rscript examples/rpart_repro.R

library(rpart)
options(digits = 12)

d <- mtcars[, c("mpg", "wt", "hp", "qsec")]
ctl <- rpart.control(cp = 0, minsplit = 4, minbucket = 2)

fit <- rpart(mpg ~ wt + hp + qsec, data = d, control = ctl)

cat("=== TREE ===\n")
print(fit)
cat("=== SPLITS (count, improve, index) ===\n")
print(fit$splits)
cat("=== FRAME (var, n, deviance, yval) ===\n")
print(data.frame(var = fit$frame$var, n = fit$frame$n,
                 dev = fit$frame$dev, yval = fit$frame$yval))
cat("=== WHERE (node id per row) ===\n")
print(fit$where)
cat("=== PRED ===\n")
print(predict(fit))
cat("=== DATA (Rust-ready) ===\n")
for (col in c("mpg", "wt", "hp", "qsec")) {
  cat(col, ": [", paste(sprintf("%.10g", d[[col]]), collapse = ", "),
      "],\n", sep = "")
}

# ---- NA variant: knock out wt for rows 2, 6, 15 (1-based) ----
d2 <- d
d2$wt[c(2, 6, 15)] <- NA
fit2 <- rpart(mpg ~ wt + hp + qsec, data = d2, control = ctl)
cat("=== NA_TREE ===\n")
print(fit2)
cat("=== NA_SPLITS ===\n")
print(fit2$splits)
cat("=== NA_WHERE ===\n")
print(fit2$where)
cat("=== NA_PRED ===\n")
print(predict(fit2))
