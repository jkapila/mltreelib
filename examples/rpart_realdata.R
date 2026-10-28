# Live rpart baseline for examples/realdata_benchmark.py.
#   Rscript examples/rpart_realdata.R <train.csv> <test.csv>
# CSVs have predictors + a final `y` column. Prints one parseable line:
#   RPART rmse=<x> mae=<x> fit=<secs>

args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(rpart))

train <- read.csv(args[1])
test <- read.csv(args[2])

t0 <- proc.time()[["elapsed"]]
fit <- rpart(y ~ ., data = train)  # default rpart control (cp = 0.01)
dt <- proc.time()[["elapsed"]] - t0

p <- predict(fit, test)
err <- p - test$y
cat(sprintf(
  "RPART rmse=%.4f mae=%.4f fit=%.3f\n",
  sqrt(mean(err^2)), mean(abs(err)), dt
))
