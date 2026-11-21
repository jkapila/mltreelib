model_matrix <- function(x, levels) {
  if (is.data.frame(x)) {
    x <- as.data.frame(Map(function(col, keys) {
      if (is.null(keys)) as.numeric(col) else match(as.character(col), keys)
    }, x, levels))
  } else {
    x <- as.matrix(x)
    for (j in which(!vapply(levels, is.null, logical(1)))) {
      x[, j] <- match(as.character(x[, j]), levels[[j]])
    }
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

#' Fit a native ModelTree
#' @param x A numeric matrix or data frame. Factor and character columns use training codes.
#' @param y A numeric response. Missing responses are ignored by the native learner.
#' @param max_depth Maximum tree depth.
#' @param min_rows Minimum rows per child leaf.
#' @return An object of class `mltreelib_model_tree`.
#' @export
fit_model_tree <- function(x, y, max_depth = 6L, min_rows = 2L) {
  levels <- if (is.data.frame(x)) lapply(x, function(col) {
    if (is.factor(col)) levels(droplevels(col))
    else if (is.character(col)) levels(factor(col))
  }) else vector("list", ncol(as.matrix(x)))
  x <- model_matrix(x, levels)
  y <- as.numeric(y)
  if (length(y) != nrow(x)) stop("x and y must have the same number of rows")
  if (!length(x) || !nrow(x)) stop("x must have at least one row and one column")
  for (control in c("max_depth", "min_rows")) {
    value <- get(control, inherits = FALSE)
    if (!is.numeric(value) || is.complex(value) || length(value) != 1L || !is.finite(value) ||
        abs(value) > .Machine$integer.max || value != trunc(value)) {
      stop(control, " must be one integer")
    }
  }
  ptr <- .Call(mltreelib_fit, x, y, as.integer(max_depth), as.integer(min_rows))
  structure(list(pointer = ptr, n_features = ncol(x), levels = levels), class = "mltreelib_model_tree")
}

#' Predict with a native ModelTree
#' @param object A fitted `mltreelib_model_tree`.
#' @param newdata A numeric matrix or data frame with the training columns in order.
#' @return A numeric vector.
#' @export
predict_model_tree <- function(object, newdata) {
  if (!inherits(object, "mltreelib_model_tree")) stop("object is not an mltreelib model")
  width <- if (is.data.frame(newdata)) ncol(newdata) else ncol(as.matrix(newdata))
  if (width != object$n_features) stop("newdata has the wrong number of columns")
  newdata <- model_matrix(newdata, object$levels)
  .Call(mltreelib_predict, object$pointer, newdata)
}

#' @export
model_tree <- fit_model_tree
