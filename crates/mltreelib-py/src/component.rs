use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use mltreelib_core::component::{
    ComponentModelCore, FactorAnalysisModel, FastIcaModel, NmfModel, NmfSolver, PcaModel, RotationMethod,
    TruncatedSvdModel, WhiteningKind, WhiteningModel,
};
use mltreelib_core::report::ModelReport;

#[pyclass(name = "ModelReport")]
#[derive(Clone)]
pub struct PyModelReport {
    pub(crate) inner: ModelReport,
}

#[pymethods]
impl PyModelReport {
    #[getter]
    fn title(&self) -> String {
        self.inner.title.clone()
    }

    fn to_text(&self) -> String {
        self.inner.to_text()
    }

    fn to_markdown(&self) -> String {
        self.inner.to_markdown()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("<ModelReport title={:?} version={}>", self.inner.title, self.inner.version)
    }

    fn __str__(&self) -> String {
        self.inner.to_text()
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let rep = ModelReport::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyModelReport { inner: rep })
    }
}

#[pyclass(name = "ComponentModel")]
pub struct PyComponentModel {
    pub(crate) inner: ComponentModelCore,
}

#[pymethods]
impl PyComponentModel {
    #[staticmethod]
    #[pyo3(signature = (
        data,
        kind = "pca",
        n_components = None,
        standardize = true,
        max_iter = None,
        tol = None,
        seed = None,
        whitening_kind = None,
        eps = None,
        rotation = None,
        promax_m = None,
        solver = None,
        l1_reg = None,
        l2_reg = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        data: Vec<Vec<f64>>,
        kind: &str,
        n_components: Option<usize>,
        standardize: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
        whitening_kind: Option<&str>,
        eps: Option<f64>,
        rotation: Option<&str>,
        promax_m: Option<f64>,
        solver: Option<&str>,
        l1_reg: Option<f64>,
        l2_reg: Option<f64>,
    ) -> PyResult<Self> {
        let k = kind.to_lowercase();
        let core = match k.as_str() {
            "pca" => {
                let rot_method = RotationMethod::parse(rotation.unwrap_or("none"), promax_m)
                    .map_err(PyValueError::new_err)?;
                let m = PcaModel::fit(&data, n_components, standardize.unwrap_or(true), Some(rot_method))
                    .map_err(PyValueError::new_err)?;
                ComponentModelCore::Pca(m)
            }
            "svd" | "truncated_svd" => {
                let m = TruncatedSvdModel::fit(&data, n_components)
                    .map_err(PyValueError::new_err)?;
                ComponentModelCore::Svd(m)
            }
            "whitening" => {
                let w_kind = match whitening_kind.unwrap_or("zca").to_lowercase().as_str() {
                    "pca" => WhiteningKind::Pca,
                    _ => WhiteningKind::Zca,
                };
                let m = WhiteningModel::fit(&data, w_kind, eps)
                    .map_err(PyValueError::new_err)?;
                ComponentModelCore::Whitening(m)
            }
            "ica" | "fastica" => {
                let m = FastIcaModel::fit(&data, n_components, max_iter, tol, seed)
                    .map_err(PyValueError::new_err)?;
                ComponentModelCore::Ica(m)
            }
            "nmf" => {
                let nmf_solver = NmfSolver::parse(solver.unwrap_or("hals"))
                    .map_err(PyValueError::new_err)?;
                let m = NmfModel::fit(
                    &data,
                    n_components,
                    Some(nmf_solver),
                    max_iter,
                    tol,
                    seed,
                    l1_reg,
                    l2_reg,
                )
                .map_err(PyValueError::new_err)?;
                ComponentModelCore::Nmf(m)
            }
            "fa" | "factor_analysis" => {
                let rot_method = RotationMethod::parse(rotation.unwrap_or("none"), promax_m)
                    .map_err(PyValueError::new_err)?;
                let m = FactorAnalysisModel::fit(
                    &data,
                    n_components,
                    max_iter,
                    tol,
                    Some(rot_method),
                    solver.unwrap_or("paf"),
                )
                .map_err(PyValueError::new_err)?;
                ComponentModelCore::Fa(m)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported component method kind: {}",
                    other
                )))
            }
        };
        Ok(PyComponentModel { inner: core })
    }

    fn transform(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.transform(&data).map_err(PyValueError::new_err)
    }

    fn inverse_transform(&self, scores: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.inverse_transform(&scores).map_err(PyValueError::new_err)
    }

    fn reconstruct(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.reconstruct(&data).map_err(PyValueError::new_err)
    }

    fn score_samples(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.inner
            .reconstruction_errors(&data)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn components(&self) -> PyResult<Vec<Vec<f64>>> {
        match &self.inner {
            ComponentModelCore::Pca(m) => Ok(m.components.clone()),
            ComponentModelCore::Svd(m) => Ok(m.components.clone()),
            ComponentModelCore::Whitening(m) => Ok(m.whitening_matrix.clone()),
            ComponentModelCore::Ica(m) => Ok(m.components.clone()),
            ComponentModelCore::Nmf(m) => Ok(m.components.clone()),
            ComponentModelCore::Fa(m) => Ok(m.components.clone()),
        }
    }

    #[getter]
    fn explained_variance(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.inner {
            ComponentModelCore::Pca(m) => Ok(Some(m.explained_variance.clone())),
            ComponentModelCore::Svd(m) => Ok(Some(m.explained_variance.clone())),
            _ => Ok(None),
        }
    }

    #[getter]
    fn explained_variance_ratio(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.inner {
            ComponentModelCore::Pca(m) => Ok(Some(m.explained_variance_ratio.clone())),
            ComponentModelCore::Svd(m) => Ok(Some(m.explained_variance_ratio.clone())),
            _ => Ok(None),
        }
    }

    #[getter]
    fn rotation_matrix(&self) -> PyResult<Option<Vec<Vec<f64>>>> {
        match &self.inner {
            ComponentModelCore::Pca(m) => Ok(m.rotation_matrix.clone()),
            ComponentModelCore::Fa(m) => Ok(m.rotation_matrix.clone()),
            _ => Ok(None),
        }
    }

    #[getter]
    fn factor_correlations(&self) -> PyResult<Option<Vec<Vec<f64>>>> {
        match &self.inner {
            ComponentModelCore::Fa(m) => Ok(m.factor_correlations.clone()),
            _ => Ok(None),
        }
    }

    #[getter]
    fn uniquenesses(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.inner {
            ComponentModelCore::Fa(m) => Ok(Some(m.uniquenesses.clone())),
            _ => Ok(None),
        }
    }

    #[getter]
    fn communalities(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.inner {
            ComponentModelCore::Fa(m) => Ok(Some(m.communalities.clone())),
            _ => Ok(None),
        }
    }

    #[allow(clippy::type_complexity)]
    fn schmid_leiman(&self) -> PyResult<Option<(Vec<Vec<f64>>, Vec<f64>)>> {
        match &self.inner {
            ComponentModelCore::Fa(m) => {
                m.schmid_leiman().map(Some).map_err(PyValueError::new_err)
            }
            _ => Ok(None),
        }
    }

    #[getter]
    fn reconstruction_error(&self) -> PyResult<Option<f64>> {
        match &self.inner {
            ComponentModelCore::Nmf(m) => Ok(Some(m.reconstruction_error)),
            _ => Ok(None),
        }
    }

    fn report(&self) -> PyModelReport {
        PyModelReport {
            inner: self.inner.report(),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let core = ComponentModelCore::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyComponentModel { inner: core })
    }
}
