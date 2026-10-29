# R-environment scorer for the Elo arena. Run FIRST:
#   R_LIBS_USER=$HOME/R/library Rscript examples/elo_score_r.R
# Reads benchmarks/elo_data/<ds>_{train,test}.csv, appends rows
# (dataset,model,rmse,mae,fit_s) to benchmarks/elo_scores.csv.

data_dir <- file.path("benchmarks", "elo_data")
out <- file.path("benchmarks", "elo_scores.csv")
# Datasets are discovered from the prepared files - add data without
# touching the scorers.
datasets <- sort(sub("_train.csv$", "",
                     list.files(data_dir, pattern = "_train.csv$")))
if (nzchar(Sys.getenv("ELO_DATASETS"))) {  # groups-of-5 runs: score a subset
  datasets <- datasets[datasets %in% strsplit(Sys.getenv("ELO_DATASETS"), ",")[[1]]]
}
done <- if (file.exists(out)) {
  read.csv(out, stringsAsFactors = FALSE)[, c("dataset", "model")]
} else {
  data.frame(dataset = character(0), model = character(0))
}
if (!file.exists(out)) {
  writeLines("dataset,model,rmse,mae,fit_s", out)
}

fit_predict <- function(model, train, test, ycol, xs) {
  # Best-effort per-model budget (algorithm x dataset): hard kills for native
  # fits go through the python-side subprocess harness; this bounds R-level work.
  setTimeLimit(elapsed = 300, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf), add = TRUE)
  switch(model,
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
    grf_llf = {
      fit <- grf::ll_regression_forest(as.matrix(train[, xs]), train[[ycol]])
      as.numeric(predict(fit, as.matrix(test[, xs]))$predictions)
    },
    grf_rf = {
      fit <- grf::regression_forest(as.matrix(train[, xs]), train[[ycol]])
      as.numeric(predict(fit, as.matrix(test[, xs]))$predictions)
    },
    ctree = {
      fit <- partykit::ctree(as.formula(paste(ycol, "~ .")), data = train)
      predict(fit, test)
    },
    glmtree = {
      f <- as.formula(paste(ycol, "~", paste0("`", xs, "`", collapse = " + "), "|",
                            paste0("`", xs, "`", collapse = " + ")))
      fit <- partykit::glmtree(f, data = train)
      predict(fit, test, type = "response")
    },
    earth = {
      fit <- earth::earth(x = train[, xs, drop = FALSE], y = train[[ycol]])
      predict(fit, test[, xs, drop = FALSE])
    },
    stop(paste("unknown model:", model))
  )
}

append_row <- function(ds, model, rmse, mae, dt) {
  # Upsert: one row per (dataset, model); rerunning is incremental.
  new <- data.frame(dataset = ds, model = model,
                    rmse = round(rmse, 4), mae = round(mae, 4),
                    fit_s = round(dt, 3))
  if (file.exists(out)) {
    old <- read.csv(out, stringsAsFactors = FALSE)
    old <- old[!(old$dataset == ds & old$model == model), ]
    new <- rbind(old, new)
  }
  write.csv(new, out, row.names = FALSE, quote = FALSE)
}

for (ds in datasets) {
  train <- read.csv(file.path(data_dir, paste0(ds, "_train.csv")), check.names = FALSE)
  test <- read.csv(file.path(data_dir, paste0(ds, "_test.csv")), check.names = FALSE)
  ycol <- "y"
  xs <- setdiff(names(train), ycol)
  for (model in c("rpart", "cubist", "lmtree", "grf_llf", "grf_rf",
                  "ctree", "glmtree", "earth")) {
    if (any(done$dataset == ds & done$model == model)) next
    cat(sprintf("%-22s %-10s RUNNING\n", ds, model)); flush(stdout())
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch({
      pred <- as.numeric(fit_predict(model, train, test, ycol, xs))
      ok <- is.finite(pred)
      if (!any(ok)) stop("no finite predictions")
      err <- pred[ok] - test[[ycol]][ok]
      if (sum(!ok) > 0) {
        message(sprintf("  %s %s: %d NA predictions dropped", ds, model, sum(!ok)))
      }
      c(sqrt(mean(err^2)), mean(abs(err)))
    }, error = function(e) {
      message(sprintf("  %s %s FAILED: %s", ds, model, conditionMessage(e)))
      NA
    })
    dt <- proc.time()[["elapsed"]] - t0
    if (length(res) == 2) {
      cat(sprintf("%-9s %-8s rmse=%8.3f mae=%8.3f fit=%6.2f\n",
                  ds, model, res[1], res[2], dt))
      append_row(ds, model, res[1], res[2], dt)
    } else {
      cat(sprintf("%-9s %-8s FAILED\n", ds, model))
    }
  }
}
cat("R scorer done ->", out, "\n")
