#include <R.h>
#include <Rinternals.h>
#include <limits.h>

extern void *mltreelib_fit_native(const double *, const double *, int, int, int, int);
extern int mltreelib_predict_native(const void *, const double *, double *, int, int);
extern void mltreelib_free_native(void *);

static SEXP model_tag;

static void finalizer(SEXP p) {
  if (R_ExternalPtrTag(p) != model_tag) return;
  void *addr = R_ExternalPtrAddr(p);
  if (addr) {
    R_ClearExternalPtr(p);
    mltreelib_free_native(addr);
  }
}

static void matrix_shape(SEXP x, int *nr, int *nc) {
  if (TYPEOF(x) != REALSXP || !Rf_isMatrix(x))
    Rf_error("data must be a double matrix");
  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (TYPEOF(dims) != INTSXP || XLENGTH(dims) != 2)
    Rf_error("invalid matrix dimensions");
  *nr = INTEGER(dims)[0];
  *nc = INTEGER(dims)[1];
  if (*nr < 0 || *nc < 0 || (*nr && *nc > INT_MAX / *nr) ||
      XLENGTH(x) != (R_xlen_t)*nr * *nc)
    Rf_error("invalid matrix dimensions for native ModelTree");
}

static int integer_control(SEXP x, int minimum, const char *name) {
  if (TYPEOF(x) != INTSXP || XLENGTH(x) != 1 || INTEGER(x)[0] < minimum)
    Rf_error("%s must be one integer greater than or equal to %d", name, minimum);
  return INTEGER(x)[0];
}

static SEXP mltreelib_fit(SEXP x, SEXP y, SEXP depth, SEXP min_rows) {
  int nr, nc;
  matrix_shape(x, &nr, &nc);
  if (!nr || !nc) Rf_error("x must have at least one row and one column");
  if (TYPEOF(y) != REALSXP || Rf_getAttrib(y, R_DimSymbol) != R_NilValue)
    Rf_error("y must be a double vector");
  if (XLENGTH(y) != nr) Rf_error("x and y must have the same number of rows");
  int depth_value = integer_control(depth, 0, "max_depth");
  int min_rows_value = integer_control(min_rows, 1, "min_rows");
  SEXP out = PROTECT(R_MakeExternalPtr(NULL, model_tag, R_NilValue));
  R_RegisterCFinalizerEx(out, finalizer, TRUE);
  void *ptr = mltreelib_fit_native(REAL(x), REAL(y), nr, nc, depth_value, min_rows_value);
  if (!ptr) Rf_error("native ModelTree fit failed");
  R_SetExternalPtrAddr(out, ptr);
  UNPROTECT(1);
  return out;
}

static SEXP mltreelib_predict(SEXP ptr, SEXP x) {
  if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != model_tag)
    Rf_error("invalid ModelTree pointer");
  const void *tree = R_ExternalPtrAddr(ptr);
  if (!tree) Rf_error("model pointer is null");
  int nr, nc;
  matrix_shape(x, &nr, &nc);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, nr));
  int rc = mltreelib_predict_native(tree, REAL(x), REAL(out), nr, nc);
  if (rc == -2) Rf_error("newdata has the wrong number of columns");
  if (rc != 0) Rf_error("native ModelTree prediction failed");
  UNPROTECT(1);
  return out;
}

static const R_CallMethodDef calls[] = {
  {"mltreelib_fit", (DL_FUNC) &mltreelib_fit, 4},
  {"mltreelib_predict", (DL_FUNC) &mltreelib_predict, 2},
  {NULL, NULL, 0}
};

void R_init_mltreelib(DllInfo *dll) {
  model_tag = Rf_install("mltreelib_model_tree");
  R_registerRoutines(dll, NULL, calls, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
