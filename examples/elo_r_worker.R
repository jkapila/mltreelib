# Fit one R (algorithm x dataset) as a secondary command-line process, run
# under the budget harness (Rscript --vanilla in its own process group).
# setTimeLimit stays as a cooperative floor (290s); the hard kill (300s) is external via SIGKILL.
#
#   Rscript --vanilla examples/elo_r_worker.R [task] <dataset> <model>
#
# prints exactly one line:
#   reg: RESULT dataset,model,rmse,mae,fit_s
#   cla: RESULT dataset,model,logloss,accuracy,bal_acc,auc,macro_f1,fit_s

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2) {
  task <- "reg"
  ds <- args[[1]]
  model <- args[[2]]
} else if (length(args) >= 3) {
  task <- args[[1]]
  ds <- args[[2]]
  model <- args[[3]]
} else {
  stop("Usage: Rscript elo_r_worker.R [task] <dataset> <model>")
}

setTimeLimit(elapsed = 290, transient = TRUE)  # cooperative floor (hard cap 300s external)

if (task %in% c("cla", "classification")) {
  data_dir <- file.path("benchmarks", "elo_cla_data")
  pred_dir <- file.path("benchmarks", "cla_preds")
  
  parse_meta <- function(path) {
    txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
    grab <- function(pattern) {
      r <- regmatches(txt, regexec(pattern, txt, perl = TRUE))[[1]]
      if (length(r) < 2) stop("meta.json: cannot find ", pattern, " in ", path)
      r[2]
    }
    target <- grab('"target"\\s*:\\s*"([^"]*)"')
    task_type <- grab('"task"\\s*:\\s*"([^"]*)"')
    cls_block <- grab('"classes"\\s*:\\s*\\[([^]]*)\\]')
    classes <- regmatches(cls_block, gregexpr('"[^"]*"', cls_block))[[1]]
    classes <- substring(classes, 2L, nchar(classes) - 1L)
    if (length(classes) < 2) stop("meta.json: no classes in ", path)
    col_block <- grab('"columns"\\s*:\\s*\\{([^}]*)\\}')
    pairs <- regmatches(col_block,
      gregexpr('"[^"]*"\\s*:\\s*"(numeric|categorical)"', col_block, perl = TRUE))[[1]]
    if (length(pairs) == 0) stop("meta.json: no columns in ", path)
    role <- vapply(pairs, function(p) {
      r <- regmatches(p, regexec('"([^"]*)"\\s*:\\s*"([^"]*)"', p, perl = TRUE))[[1]]
      c(r[2], r[3])
    }, character(2))
    list(target = target, task = task_type, classes = classes,
         role = stats::setNames(role[2, ], role[1, ]))
  }

  meta <- parse_meta(file.path(data_dir, paste0(ds, ".meta.json")))
  classes <- meta$classes
  K <- length(classes)

  # Read train and test
  train_raw <- read.csv(file.path(data_dir, paste0(ds, "_train.csv")),
                        check.names = FALSE, colClasses = "character",
                        stringsAsFactors = FALSE, na.strings = character(0))
  test_raw <- read.csv(file.path(data_dir, paste0(ds, "_test.csv")),
                       check.names = FALSE, colClasses = "character",
                       stringsAsFactors = FALSE, na.strings = character(0))

  ycol <- meta$target
  cats <- names(meta$role)[meta$role == "categorical"]
  nums <- setdiff(names(meta$role), cats)

  train <- train_raw
  test <- test_raw

  for (col in cats) {
    lev <- sort(unique(c(train_raw[[col]], test_raw[[col]])))
    train[[col]] <- factor(train_raw[[col]], levels = lev)
    test[[col]] <- factor(test_raw[[col]], levels = lev)
  }
  for (col in nums) {
    train[[col]] <- suppressWarnings(as.numeric(train_raw[[col]]))
    test[[col]] <- suppressWarnings(as.numeric(test_raw[[col]]))
  }
  train[[ycol]] <- factor(as.character(train_raw[[ycol]]), levels = classes)
  test[[ycol]] <- factor(as.character(test_raw[[ycol]]), levels = classes)

  xs <- setdiff(names(train), ycol)

  bmat <- function(p) {
    p <- as.numeric(p)
    m <- cbind(1 - p, p)
    colnames(m) <- classes
    m
  }

  t0 <- proc.time()[["elapsed"]]
  pred_prob <- switch(model,
    rpart = {
      fit <- rpart::rpart(as.formula(paste0("`", ycol, "` ~ .")), data = train, method = "class")
      predict(fit, test, type = "prob")
    },
    grf_llf = ,
    grf_rf = {
      xt <- train[, xs, drop = FALSE]
      if (!all(vapply(xt, is.numeric, TRUE))) {
        stop("grf probability_forest requires purely numeric features (no factors)")
      }
      fit <- grf::probability_forest(as.matrix(xt), train[[ycol]])
      predict(fit, as.matrix(test[, xs, drop = FALSE]))$predictions
    },
    ctree = {
      fit <- partykit::ctree(as.formula(paste0("`", ycol, "` ~ .")), data = train)
      predict(fit, test, type = "prob")
    },
    glmtree = {
      if (K != 2) stop("glmtree is binary-only")
      f <- as.formula(paste0("`", ycol, "` ~ ",
                             paste0("`", xs, "`", collapse = " + "), " | ",
                             paste0("`", xs, "`", collapse = " + ")))
      fit <- partykit::glmtree(f, data = train, family = binomial)
      bmat(predict(fit, test, type = "response"))
    },
    earth = {
      if (K != 2) stop("earth is binary-only")
      fit <- earth::earth(x = train[, xs, drop = FALSE],
                          y = as.numeric(train[[ycol]] == classes[2]),
                          glm = list(family = binomial))
      bmat(predict(fit, test[, xs, drop = FALSE], type = "response"))
    },
    mboost = {
      if (K == 2) {
        fit <- mboost::blackboost(as.formula(paste0("`", ycol, "` ~ .")),
                                  data = train, family = mboost::Binomial())
        bmat(predict(fit, test, type = "response"))
      } else {
        cls <- "..class"
        while (cls %in% c(xs, ycol)) cls <- paste0(".", cls)
        f <- as.formula(paste0("`", ycol, "` ~ ",
          paste0("bols(`", xs, "`, df = 2) %O% bols(`", cls,
                 "`, df = 2, contrasts.arg = \"contr.dummy\")", collapse = " + ")))
        environment(f) <- list2env(list(`%O%` = mboost::`%O%`, bols = mboost::bols),
                                   parent = environment())
        tr <- as.list(train[, xs, drop = FALSE]); tr[[ycol]] <- train[[ycol]]
        tr[[cls]] <- factor(classes[-K])
        te <- as.list(test[, xs, drop = FALSE]); te[[cls]] <- factor(classes[-K])
        fit <- mboost::mboost(f, data = tr, family = mboost::Multinomial(),
                              baselearner = mboost::bbs)
        predict(fit, newdata = te, type = "response")
      }
    },
    ranger = {
      fit <- ranger::ranger(as.formula(paste0("`", ycol, "` ~ .")), data = train,
                            probability = TRUE)
      predict(fit, test)$predictions
    },
    bart = {
      if (K != 2) stop("bart is binary-only")
      train_b <- train
      train_b[[ycol]] <- as.numeric(train[[ycol]] == classes[2])
      test_b <- test
      test_b[[ycol]] <- NULL
      fit <- dbarts::bart2(as.formula(paste0("`", ycol, "` ~ .")), data = train_b,
                           test = test_b, verbose = FALSE)
      yh <- pnorm(fit$yhat.test)
      bmat(if (length(dim(yh)) == 3L) apply(yh, 3, mean) else colMeans(yh))
    },
    c50_tree = {
      xtr <- train[, xs, drop = FALSE]
      xte <- test[, xs, drop = FALSE]
      for (col in cats) {
        if (col %in% xs) {
          lvls <- levels(xtr[[col]])
          clean <- paste0("L", seq_along(lvls))
          levels(xtr[[col]]) <- clean
          levels(xte[[col]]) <- clean
        }
      }
      fit <- C50::C5.0(x = xtr, y = train[[ycol]], trials = 1)
      predict(fit, xte, type = "prob")
    },
    c50_rules = {
      xtr <- train[, xs, drop = FALSE]
      xte <- test[, xs, drop = FALSE]
      for (col in cats) {
        if (col %in% xs) {
          lvls <- levels(xtr[[col]])
          clean <- paste0("L", seq_along(lvls))
          levels(xtr[[col]]) <- clean
          levels(xte[[col]]) <- clean
        }
      }
      fit <- C50::C5.0(x = xtr, y = train[[ycol]], rules = TRUE)
      predict(fit, xte, type = "prob")
    },
    stop("unknown model: ", model)
  )
  dt <- proc.time()[["elapsed"]] - t0

  # Align probability columns
  if (is.list(pred_prob) && !is.matrix(pred_prob)) pred_prob <- do.call(rbind, pred_prob)
  proba <- as.matrix(pred_prob)
  storage.mode(proba) <- "double"
  if (!is.null(colnames(proba))) {
    if (!all(classes %in% colnames(proba))) stop("prediction columns do not cover meta classes")
    proba <- proba[, classes, drop = FALSE]
  } else {
    if (ncol(proba) != length(classes)) stop("prediction width does not match meta classes")
    colnames(proba) <- classes
  }

  y_true <- as.character(test[[ycol]])
  y_idx <- match(y_true, classes)
  n <- length(y_idx)
  am <- max.col(proba, ties.method = "first")
  p_true <- pmin(pmax(proba[cbind(seq_len(n), y_idx)], 1e-15), 1 - 1e-15)
  logloss <- -mean(log(p_true))
  accuracy <- mean(am == y_idx)
  present <- sort(unique(y_idx))
  rec <- vapply(present, function(c) mean(am[y_idx == c] == c), numeric(1))
  bal_acc <- mean(rec)
  auc <- NA_real_
  if (ncol(proba) == 2) {
    pos <- y_idx == 2L
    n_pos <- sum(pos); n_neg <- n - n_pos
    if (n_pos > 0 && n_neg > 0) {
      r <- rank(proba[, 2], ties.method = "average")
      u <- sum(r[pos]) - n_pos * (n_pos + 1) / 2
      auc <- u / (n_pos * n_neg)
    }
  }
  f1s <- vapply(seq_len(ncol(proba)), function(c) {
    tp <- sum(am == c & y_idx == c)
    fp <- sum(am == c & y_idx != c)
    fn <- sum(am != c & y_idx == c)
    prec <- if (tp + fp > 0) tp / (tp + fp) else 0
    rec_c <- if (tp + fn > 0) tp / (tp + fn) else 0
    if (prec + rec_c > 0) 2 * prec * rec_c / (prec + rec_c) else 0
  }, numeric(1))
  macro_f1 <- sum(f1s) / max(1, ncol(proba))

  # Write predictions dump
  if (!dir.exists(pred_dir)) dir.create(pred_dir, recursive = TRUE)
  d_pred <- as.data.frame(proba, check.names = FALSE)
  d_pred$pred <- classes[am]
  d_pred$y <- y_true
  write.csv(d_pred, file.path(pred_dir, paste0(ds, "__", model, ".csv")),
            row.names = FALSE, quote = FALSE)

  auc_str <- if (is.na(auc)) "" else sprintf("%.4f", auc)
  cat(sprintf("RESULT %s,%s,%.4f,%.4f,%.4f,%s,%.4f,%.3f\n",
              ds, model, logloss, accuracy, bal_acc, auc_str, macro_f1, dt))

} else {
  # Regression task
  data_dir <- file.path("benchmarks", "elo_data")
  train <- read.csv(file.path(data_dir, paste0(ds, "_train.csv")),
                    check.names = FALSE, stringsAsFactors = TRUE)
  test <- read.csv(file.path(data_dir, paste0(ds, "_test.csv")),
                   check.names = FALSE, stringsAsFactors = TRUE)
  ycol <- "y"
  xs <- setdiff(names(train), ycol)

  train_mat <- data.matrix(train[, xs, drop = FALSE])
  test_mat <- data.matrix(test[, xs, drop = FALSE])

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
      fit <- grf::ll_regression_forest(train_mat, train[[ycol]])
      predict(fit, test_mat)$predictions
    },
    grf_rf = {
      fit <- grf::regression_forest(train_mat, train[[ycol]])
      predict(fit, test_mat)$predictions
    },
    earth = {
      fit <- earth::earth(as.formula(paste(ycol, "~ .")), data = train)
      predict(fit, test)
    },
    mboost = {
      fit <- mboost::blackboost(as.formula(paste(ycol, "~ .")), data = train)
      predict(fit, test)
    },
    ranger = {
      fit <- ranger::ranger(as.formula(paste(ycol, "~ .")), data = train)
      predict(fit, test)$predictions
    },
    bart = {
      fit <- dbarts::bart(x.train = train_mat,
                          y.train = train[[ycol]],
                          x.test = test_mat,
                          verbose = FALSE)
      colMeans(fit$yhat.test)
    },
    fru = {
      fit <- fru::fru(train[, xs, drop = FALSE], train[[ycol]])
      predict(fit, test[, xs, drop = FALSE])
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
}
