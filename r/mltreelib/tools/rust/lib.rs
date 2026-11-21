use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{mem::size_of, ptr, slice};

fn rows(x: &[f64], nr: usize, nc: usize) -> Vec<Vec<Option<f64>>> {
    (0..nr)
        .map(|i| {
            (0..nc)
                .map(|j| {
                    let value = x[i + j * nr];
                    (!value.is_nan()).then_some(value)
                })
                .collect()
        })
        .collect()
}

fn matrix_len(nr: i32, nc: i32) -> Option<usize> {
    let len = usize::try_from(nr)
        .ok()?
        .checked_mul(usize::try_from(nc).ok()?)?;
    (len <= isize::MAX as usize / size_of::<f64>()).then_some(len)
}

#[no_mangle]
pub unsafe extern "C" fn mltreelib_fit_native(
    x: *const f64,
    y: *const f64,
    nr: i32,
    nc: i32,
    depth: i32,
    min_rows: i32,
) -> *mut ModelTree {
    if x.is_null() || y.is_null() || nr <= 0 || nc <= 0 || depth < 0 || min_rows < 1 {
        return ptr::null_mut();
    }
    let result = catch_unwind(|| {
        let x = slice::from_raw_parts(x, matrix_len(nr, nc)?);
        let y = slice::from_raw_parts(y, nr as usize);
        let target = y
            .iter()
            .map(|&v| (!v.is_nan()).then_some(v))
            .collect::<Vec<_>>();
        let params = ModelTreeParams {
            max_depth: depth as usize,
            min_rows: min_rows as usize,
            ..ModelTreeParams::default()
        };
        ModelTree::fit(&rows(x, nr as usize, nc as usize), &target, params)
            .ok()
            .map(|tree| Box::into_raw(Box::new(tree)))
    });
    result.ok().flatten().unwrap_or(ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn mltreelib_predict_native(
    tree: *const ModelTree,
    x: *const f64,
    out: *mut f64,
    nr: i32,
    nc: i32,
) -> i32 {
    let Some(len) = matrix_len(nr, nc) else {
        return -1;
    };
    if tree.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        if (*tree).col_names.len() != nc as usize {
            return -2;
        }
        if nr == 0 {
            return 0;
        }
        if x.is_null() || out.is_null() {
            return -1;
        }
        let x = slice::from_raw_parts(x, len);
        let out = slice::from_raw_parts_mut(out, nr as usize);
        out.copy_from_slice(&(*tree).predict(&rows(x, nr as usize, nc as usize)));
        0
    }))
    .unwrap_or(-3)
}

#[no_mangle]
pub unsafe extern "C" fn mltreelib_free_native(tree: *mut ModelTree) {
    if !tree.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(tree))));
    }
}
