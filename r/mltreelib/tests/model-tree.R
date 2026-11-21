library(mltreelib)

expect_error <- function(expr, pattern) {
  result <- tryCatch(expr, error = conditionMessage)
  stopifnot(is.character(result), grepl(pattern, result, fixed = TRUE))
}

x <- cbind(a = c(1, 2, 3, 4, NA, 6), b = c(0, 1, 0, 1, 0, 1))
y <- c(1, 2, 3, 4, 5, 6)
fit <- fit_model_tree(x, y, max_depth = 2L, min_rows = 1L)
pred <- predict_model_tree(fit, x)
stopifnot(length(pred) == nrow(x), all(is.finite(pred)))
stopifnot(identical(predict_model_tree(fit, x[FALSE, ]), numeric()))
expect_error(predict_model_tree(fit, x[, 1, drop = FALSE]), "wrong number")

fit_categories <- fit_model_tree(data.frame(g = factor(c("a", "b", "a", "b"))),
                                 c(1, 2, 1, 2), min_rows = 1L)
p_categories <- predict_model_tree(fit_categories, data.frame(g = c("a", "b", NA)))
stopifnot(length(p_categories) == 3L, p_categories[1] != p_categories[2])
stopifnot(identical(p_categories[2], predict_model_tree(fit_categories, data.frame(g = factor("b")))))
stopifnot(identical(p_categories[3], predict_model_tree(fit_categories, data.frame(g = "unknown"))))
expect_error(fit_model_tree(x, y, max_depth = -1L), "max_depth")
expect_error(fit_model_tree(x, y, max_depth = 1.5), "max_depth")
expect_error(fit_model_tree(x, y, min_rows = 0L), "min_rows")

native <- getDLLRegisteredRoutines("mltreelib")$.Call
fit_native <- native[["mltreelib_fit"]]
predict_native <- native[["mltreelib_predict"]]
expect_error(.Call(fit_native, x, y, NA_integer_, 1L), "max_depth")
expect_error(.Call(fit_native, x, y, 2, 1L), "max_depth")
expect_error(.Call(fit_native, x, y, 2L, integer()), "min_rows")
expect_error(.Call(fit_native, as.numeric(x), y, 2L, 1L), "double matrix")
expect_error(.Call(fit_native, x, matrix(y), 2L, 1L), "double vector")
expect_error(.Call(fit_native, x, y[-1], 2L, 1L), "same number")
expect_error(.Call(fit_native, x, rep(NA_real_, length(y)), 2L, 1L), "native ModelTree fit failed")
expect_error(.Call(predict_native, fit$pointer, x[, 1, drop = FALSE]), "wrong number")
expect_error(.Call(predict_native, fit$pointer, as.numeric(x)), "double matrix")
expect_error(.Call(predict_native, methods::new("externalptr"), x), "invalid ModelTree pointer")
expect_error(predict_model_tree(unserialize(serialize(fit, NULL)), x), "model pointer is null")
rm(fit, fit_categories)
invisible(gc())
invisible(gc())

dat <- datasets::airquality
train <- which(dat$Month <= 7 & !is.na(dat$Ozone))
test <- which(dat$Month == 8 & !is.na(dat$Ozone))
features <- c("Solar.R", "Wind", "Temp", "Month", "Day")
model <- fit_model_tree(dat[train, features], dat$Ozone[train])
p <- predict_model_tree(model, dat[test, features])
stopifnot(length(p) == length(test), all(is.finite(p)))
