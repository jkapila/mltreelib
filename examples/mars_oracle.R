# R-earth oracle capture for the Rust MARS gates (see
# references/research/mars-port-plan.md). Fits earth on each shared fixture
# with PINNED parameters and exports everything the fixed-design gate needs:
# selected terms (dirs + cuts), coefficients, RSS, GCV, fitted values.
#
#   R_LIBS_USER=$HOME/R/library Rscript examples/mars_oracle.R
#
# Output: benchmarks/mars_fixtures/oracle/<fixture>_deg<k>/{coefs,dirs,cuts,preds}.csv + meta.txt

out_root <- file.path("benchmarks", "mars_fixtures", "oracle")
fx <- sort(list.files(file.path("benchmarks", "mars_fixtures"),
                      pattern = "\\.csv$", full.names = TRUE))

for (f in fx) {
  name <- sub("\\.csv$", "", basename(f))
  d <- read.csv(f, check.names = FALSE)
  ycol <- "y"
  xs <- setdiff(names(d), ycol)
  x <- as.matrix(d[, xs, drop = FALSE])
  y <- d[[ycol]]

  for (deg in c(1, 2)) {
    # Pinned oracle configuration: no Fast MARS (validated separately),
    # fixed penalty/thresh. meta.txt records the pins and the package version.
    fit <- earth::earth(x, y,
                        degree = deg,
                        penalty = 2,
                        thresh = 1e-4,
                        fast.k = 0,        # disables Fast MARS entirely
                        Scale.y = FALSE)   # coefficients on the original y scale

    out <- file.path(out_root, paste0(name, "_deg", deg))
    dir.create(out, recursive = TRUE, showWarnings = FALSE)
    sel <- fit$selected.terms

    write.csv(fit$coefficients, file.path(out, "coefs.csv"), row.names = FALSE)
    # full matrices + selection indices (robust for degenerate fixtures)
    write.csv(fit$dirs, file.path(out, "dirs.csv"), row.names = FALSE)
    write.csv(fit$cuts, file.path(out, "cuts.csv"), row.names = FALSE)
    write.csv(data.frame(selected_terms = sel),
              file.path(out, "selected_terms.csv"), row.names = FALSE)
    write.csv(data.frame(pred = as.numeric(fitted(fit))),
              file.path(out, "preds.csv"), row.names = FALSE)

    writeLines(c(
      sprintf("earth_version=%s", as.character(utils::packageVersion("earth"))),
      sprintf("fixture=%s", name),
      sprintf("degree=%d", deg),
      "penalty=2",
      "thresh=1e-4",
      "fast.k=0",
      "fast.beta=unused (fast.k=0 disables fast search)",
      "Scale.y=FALSE",
      sprintf("rss=%.12g", fit$rss),
      sprintf("gcv=%.12g", fit$gcv),
      sprintf("n_terms=%d", length(sel))
    ), file.path(out, "meta.txt"))
    cat(sprintf("%-24s deg=%d terms=%2d rss=%.5g\n", name, deg, length(sel), fit$rss))
  }
}
cat("oracle written ->", out_root, "\n")
