# Export R-bundled classic regression datasets into the Elo arena.
#   R_LIBS_USER=$HOME/R/library Rscript examples/elo_dump_datasets.R
# Writes <name>_{train,test}.csv into benchmarks/elo_data using the same
# 80/20 seed-7 split and full one-hot encoding as examples/elo_prepare.py.
# Factors -> one-hot columns named <col>_<level>; missing values -> empty.

data_dir <- file.path("benchmarks", "elo_data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

dump <- function(name, x, y) {
  df <- as.data.frame(x, stringsAsFactors = TRUE)
  for (n in names(df)) {
    if (is.factor(df[[n]]) || is.character(df[[n]])) {
      f <- as.factor(df[[n]])
      lv <- levels(f)
      for (l in lv) df[[paste0(n, "_", l)]] <- as.numeric(f == l)
      df[[n]] <- NULL
    }
  }
  df$y <- as.numeric(y)
  set.seed(7)
  idx <- sample(seq_len(nrow(df)))
  n_tr <- floor(0.8 * length(idx))
  tr <- sort(idx[seq_len(n_tr)])
  te <- sort(idx[(n_tr + 1):length(idx)])
  write.csv(df[tr, ], file.path(data_dir, paste0(name, "_train.csv")),
            row.names = FALSE, na = "")
  write.csv(df[te, ], file.path(data_dir, paste0(name, "_test.csv")),
            row.names = FALSE, na = "")
  cat(sprintf("%-12s %5d train / %4d test, %2d cols, %d missing cells\n",
              name, length(tr), length(te), ncol(df) - 1,
              sum(is.na(df[, setdiff(names(df), "y")]))))
}

# airquality: predict Ozone from Solar.R/Wind/Temp/Month/Day (NA-rich).
data(airquality)
dump("airquality", airquality[, c("Solar.R", "Wind", "Temp", "Month", "Day")],
     airquality$Ozone)

# mlbench::Servo: response time from motor/screw/gain (2 factors).
library(mlbench)
data(Servo)
dump("servo", Servo[, c("motor", "screw", "pgain", "vgain")], Servo$class)

# mlbench::Puma8NH / Puma32NH: large kinematics regression (8 resp. 32 cols).
data(Puma8NH)
dump("puma8nh", Puma8NH[, setdiff(names(Puma8NH), "y")], Puma8NH$y)
data(Puma32NH)
dump("puma32nh", Puma32NH[, setdiff(names(Puma32NH), "y")], Puma32NH$y)

cat("R dataset dump done ->", data_dir, "\n")
