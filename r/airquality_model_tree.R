# Use a fixed calendar split and keep the original missing Solar.R values.
library(mltreelib)

dat <- datasets::airquality
eligible <- which(!is.na(dat$Ozone))
train <- eligible[dat$Month[eligible] <= 7]
test <- eligible[dat$Month[eligible] == 8]
features <- c("Solar.R", "Wind", "Temp", "Month", "Day")
x_train <- dat[train, features]
x_test <- dat[test, features]
y_train <- dat$Ozone[train]
y_test <- dat$Ozone[test]
stopifnot(!anyNA(y_train), !anyNA(y_test), anyNA(x_train$Solar.R),
          anyNA(x_test$Solar.R), max(train) < min(test))

fit_times <- numeric(5)
predict_times <- numeric(5)
predictions <- vector("list", length(fit_times))
for (i in seq_along(fit_times)) {
  fit_times[i] <- system.time({
    fit <- fit_model_tree(x_train, y_train, max_depth = 6L, min_rows = 2L)
  }, gcFirst = FALSE)[["elapsed"]]
  predict_times[i] <- system.time({
    predictions[[i]] <- predict_model_tree(fit, x_test)
  }, gcFirst = FALSE)[["elapsed"]]
}
pred <- predictions[[1L]]
stopifnot(length(pred) == length(y_test), all(is.finite(pred)),
          all(vapply(predictions, identical, logical(1), y = pred)))

metrics <- function(pred) {
  error <- pred - y_test
  c(mae = mean(abs(error)), rmse = sqrt(mean(error^2)))
}
scores <- metrics(pred)
baseline <- metrics(mean(y_train))
stopifnot(all(is.finite(c(scores, baseline))))

cat(sprintf(paste0(
  "dataset=R datasets::airquality rows=%d eligible=%d train=%d test=%d ",
  "train_months=5-7 test_month=8 core_seed=42 max_depth=6 min_rows=2 ",
  "solar_r_na_train=%d solar_r_na_test=%d mae=%.10f rmse=%.10f ",
  "train_mean_mae=%.10f train_mean_rmse=%.10f ",
  "fit_seconds=%s predict_seconds=%s fit_median=%.6f predict_median=%.6f\n"),
  nrow(dat), length(eligible), length(train), length(test),
  sum(is.na(x_train$Solar.R)), sum(is.na(x_test$Solar.R)),
  scores[["mae"]], scores[["rmse"]], baseline[["mae"]], baseline[["rmse"]],
  paste(sprintf("%.6f", fit_times), collapse = ","),
  paste(sprintf("%.6f", predict_times), collapse = ","),
  median(fit_times), median(predict_times)))
