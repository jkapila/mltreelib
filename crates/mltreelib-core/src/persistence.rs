//! Versioned snapshots for native model state.

use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::str::Utf8Error;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{criteria::GainFn, model_tree::ModelTree, node_stats::LeafModel};

const MODEL_TREE_FORMAT: &str = "mltreelib.model_tree";
const FORMAT_VERSION: u32 = 1;

mod unsupervised;

/// An error while reading or writing a model snapshot.
#[derive(Debug)]
pub enum PersistenceError {
    Io(io::Error),
    Json(serde_json::Error),
    Utf8(Utf8Error),
    Format(&'static str),
    Version(u32),
    Invalid(&'static str),
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => err.fmt(f),
            Self::Json(err) => write!(f, "invalid model JSON: {err}"),
            Self::Utf8(err) => write!(f, "model bytes must contain UTF-8 JSON: {err}"),
            Self::Format(format) => write!(f, "expected a {format} model snapshot"),
            Self::Version(version) => {
                write!(f, "unsupported model snapshot format version {version}")
            }
            Self::Invalid(reason) => f.write_str(reason),
        }
    }
}

impl Error for PersistenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Utf8(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ModelSnapshot<F, T> {
    format: F,
    format_version: u32,
    model: T,
}

pub(crate) fn encode_snapshot<T: Serialize>(
    format: &'static str,
    model: &T,
) -> Result<String, PersistenceError> {
    serde_json::to_string_pretty(&ModelSnapshot {
        format,
        format_version: FORMAT_VERSION,
        model,
    })
    .map_err(PersistenceError::Json)
}

pub(crate) fn decode_snapshot<T: DeserializeOwned>(
    format: &'static str,
    data: &str,
) -> Result<T, PersistenceError> {
    // Check the format before asking Serde to decode the version-specific model.
    #[derive(Deserialize)]
    struct Header {
        format: String,
        format_version: u32,
    }
    let header: Header = serde_json::from_str(data).map_err(PersistenceError::Json)?;
    if header.format != format {
        return Err(PersistenceError::Format(format));
    }
    if header.format_version != FORMAT_VERSION {
        return Err(PersistenceError::Version(header.format_version));
    }
    let snapshot: ModelSnapshot<String, T> =
        serde_json::from_str(data).map_err(PersistenceError::Json)?;
    Ok(snapshot.model)
}

impl ModelTree {
    /// Serialize the fitted tree and its training configuration as versioned JSON.
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        self.check_snapshot()?;
        let mut gain = &self.params.gain;
        loop {
            match gain {
                GainFn::Custom(_) => {
                    return Err(PersistenceError::Invalid(
                        "custom split criteria cannot be serialized",
                    ));
                }
                GainFn::FairConstrained { base_gain, .. } => gain = base_gain,
                _ => break,
            }
        }
        encode_snapshot(MODEL_TREE_FORMAT, self)
    }

    /// Restore a fitted tree from a native JSON snapshot.
    pub fn from_json(data: &str) -> Result<Self, PersistenceError> {
        let model: Self = decode_snapshot(MODEL_TREE_FORMAT, data)?;
        model.check_snapshot()?;
        Ok(model)
    }

    /// Return the JSON snapshot as UTF-8 bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistenceError> {
        self.to_json().map(String::into_bytes)
    }

    /// Restore a fitted tree from UTF-8 JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, PersistenceError> {
        Self::from_json(std::str::from_utf8(data).map_err(PersistenceError::Utf8)?)
    }

    /// Save the JSON snapshot, replacing the destination only after a complete write.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        let data = self.to_json()?;
        write_snapshot(path.as_ref(), data.as_bytes()).map_err(PersistenceError::Io)
    }

    /// Read a JSON snapshot from disk and restore the fitted tree.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        let data = fs::read(path).map_err(PersistenceError::Io)?;
        Self::from_bytes(&data)
    }

    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        let width = self.col_names.len();
        let valid_cols = |cols: &[usize]| cols.iter().all(|&index| index < width);
        if !self.category_levels.is_empty()
            && (self.category_levels.len() != width
                || self
                    .category_levels
                    .iter()
                    .flatten()
                    .flat_map(|map| map.values())
                    .any(|code| !code.is_finite()))
        {
            return Err(PersistenceError::Invalid(
                "invalid model category vocabulary",
            ));
        }
        if !valid_cols(&self.active_cols)
            || self.sur_parts.iter().flatten().any(|s| s.col >= width)
            || self.oblique_parts.iter().any(|projection| {
                projection.cols.len() != projection.weights.len() || !valid_cols(&projection.cols)
            })
        {
            return Err(PersistenceError::Invalid("invalid model columns"));
        }
        if self.training_memory.as_ref().is_some_and(|memory| {
            let rows = memory.x.len();
            memory.y.len() != rows
                || memory.base_preds.len() != rows
                || memory.col_stds.len() != width
                || memory.x.iter().any(|row| row.len() != width)
        }) {
            return Err(PersistenceError::Invalid("invalid model training memory"));
        }

        let mut seen = vec![false; self.nodes.len()];
        let mut leaf_ids = vec![false; self.nodes.len()];
        let mut todo = vec![(self.root, 0usize)];
        while let Some((index, depth)) = todo.pop() {
            // Traversal is iterative here. Rule extraction and TreeSHAP recurse.
            if depth > 512 || index >= seen.len() || std::mem::replace(&mut seen[index], true) {
                return Err(PersistenceError::Invalid("invalid model tree topology"));
            }
            let node = &self.nodes[index];
            match (node.split.as_ref(), node.left, node.right) {
                (Some(split), Some(left), Some(right)) => {
                    if split.col >= width
                        || split.cat.is_some_and(|i| i >= self.cat_parts.len())
                        || split.sur.is_some_and(|i| i >= self.sur_parts.len())
                        || split.oblique.is_some_and(|i| i >= self.oblique_parts.len())
                    {
                        return Err(PersistenceError::Invalid("invalid model split"));
                    }
                    todo.extend([(left, depth + 1), (right, depth + 1)]);
                }
                (None, None, None) if node.leaf.is_some() => {
                    let id = node
                        .leaf_id
                        .ok_or(PersistenceError::Invalid("missing model leaf ID"))?;
                    if id >= leaf_ids.len() || std::mem::replace(&mut leaf_ids[id], true) {
                        return Err(PersistenceError::Invalid("invalid model leaf ID"));
                    }
                }
                _ => return Err(PersistenceError::Invalid("invalid model tree node")),
            }
            if node
                .leaf
                .as_ref()
                .is_some_and(|leaf| !valid_leaf(leaf, width))
            {
                return Err(PersistenceError::Invalid("invalid model leaf"));
            }
        }
        Ok(())
    }
}

pub(crate) fn write_snapshot(path: &Path, data: &[u8]) -> io::Result<()> {
    static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temp, mut file) = loop {
        let id = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let name = format!(".mltreelib-{}-{id}.tmp", std::process::id());
        if path.file_name() == Some(name.as_ref()) {
            continue;
        }
        let temp = parent.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temp) {
            Ok(file) => break (temp, file),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    };
    let result = file.write_all(data).and_then(|()| file.sync_all());
    drop(file);
    if let Err(err) = result.and_then(|()| fs::rename(&temp, path)) {
        let _ = fs::remove_file(temp);
        return Err(err);
    }
    #[cfg(unix)]
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn valid_leaf(leaf: &LeafModel, width: usize) -> bool {
    use crate::node_stats::{AftModel, LinearModel, SurvivalStats};

    let cols = |chosen: &[usize]| chosen.iter().all(|&index| index < width);
    let coefficient_shape =
        |chosen: &[usize], coef: &[f64]| chosen.len() == coef.len() && cols(chosen);
    let linear = |model: &LinearModel| coefficient_shape(&model.cols, &model.coef);
    let aft = |model: &AftModel| coefficient_shape(&model.cols, &model.coef);
    let survival = |stats: &SurvivalStats| {
        let times = stats.times.len();
        stats.cum_hazard.len() == times
            && stats.survival.len() == times
            && stats.causes.len() == stats.cif.len()
            && stats.causes.len() == stats.event_counts.len()
            && stats.cif.iter().all(|curve| curve.len() == times)
            && stats.times.iter().all(|time| time.is_finite())
            && stats.times.windows(2).all(|pair| pair[0] <= pair[1])
    };
    leaf.linear.as_ref().is_none_or(linear)
        && leaf.aft.as_ref().is_none_or(aft)
        && leaf.survival.as_ref().is_none_or(survival)
        && leaf.cure.as_ref().is_none_or(|model| {
            model.aft.as_ref().is_none_or(aft) && model.survival.as_ref().is_none_or(survival)
        })
        && leaf
            .spline
            .as_ref()
            .is_none_or(|model| coefficient_shape(&model.cols, &model.coef))
        && leaf
            .heteroscedastic
            .as_ref()
            .is_none_or(|model| model.scale_linear.as_ref().is_none_or(linear))
        && leaf.softmax.as_ref().is_none_or(|model| {
            model.probs.len() == model.n_classes
                && model.linear_models.as_ref().is_none_or(|models| {
                    models.len() == model.n_classes && models.iter().all(linear)
                })
        })
        && leaf.mars_cols.as_ref().is_none_or(|chosen| {
            cols(chosen)
                && leaf.mars.as_ref().is_some_and(|model| {
                    model
                        .basis
                        .iter()
                        .flat_map(|basis| &basis.factors)
                        .all(|factor| factor.feature < chosen.len())
                })
        })
        && leaf.knn.as_ref().is_none_or(|model| {
            let n = model.samples.len();
            let p = model.chosen_cols.len();
            model.k > 0
                && cols(&model.chosen_cols)
                && model.samples.iter().all(|row| row.len() == p)
                && model.feature_scales.len() == p
                && model.targets.len() == n
                && model.weights.len() == n
                && model.validity_scores.len() == n
                && model.compactness_scores.len() == n
                && model.fallback_probs.len() == model.n_classes
        })
}

/// JSON numbers hold finite values. Explicit strings preserve non-finite
/// statistics such as an unobserved median survival time.
pub(crate) mod float {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub trait FloatState: Sized {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
    }

    pub fn serialize<T: FloatState, S: Serializer>(
        value: &T,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.write(serializer)
    }

    pub fn deserialize<'de, T: FloatState, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<T, D::Error> {
        T::read(deserializer)
    }

    struct Ref<'a, T>(&'a T);

    impl<T: FloatState> Serialize for Ref<'_, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.write(serializer)
        }
    }

    struct Owned<T>(T);

    impl<'de, T: FloatState> Deserialize<'de> for Owned<T> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            T::read(deserializer).map(Self)
        }
    }

    impl FloatState for f64 {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if self.is_finite() {
                return serializer.serialize_f64(*self);
            }
            serializer.serialize_str(if self.is_nan() {
                "NaN"
            } else if self.is_sign_positive() {
                "Infinity"
            } else {
                "-Infinity"
            })
        }

        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            enum Sentinel {
                NaN,
                Infinity,
                #[serde(rename = "-Infinity")]
                NegativeInfinity,
            }
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum NumberOrSentinel {
                Number(f64),
                Sentinel(Sentinel),
            }
            match NumberOrSentinel::deserialize(deserializer)? {
                NumberOrSentinel::Number(value) => Ok(value),
                NumberOrSentinel::Sentinel(Sentinel::NaN) => Ok(f64::NAN),
                NumberOrSentinel::Sentinel(Sentinel::Infinity) => Ok(f64::INFINITY),
                NumberOrSentinel::Sentinel(Sentinel::NegativeInfinity) => Ok(f64::NEG_INFINITY),
            }
        }
    }

    impl<T: FloatState> FloatState for Option<T> {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.as_ref().map(Ref).serialize(serializer)
        }

        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Option::<Owned<T>>::deserialize(deserializer)
                .map(|value| value.map(|Owned(value)| value))
        }
    }

    impl<T: FloatState> FloatState for Vec<T> {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter().map(Ref))
        }

        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Vec::<Owned<T>>::deserialize(deserializer)
                .map(|items| items.into_iter().map(|Owned(value)| value).collect())
        }
    }

    impl FloatState for (usize, f64, f64) {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            (self.0, Ref(&self.1), Ref(&self.2)).serialize(serializer)
        }

        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            <(usize, Owned<f64>, Owned<f64>)>::deserialize(deserializer)
                .map(|(col, Owned(stat), Owned(p))| (col, stat, p))
        }
    }

    impl FloatState for (usize, f64) {
        fn write<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            (self.0, Ref(&self.1)).serialize(serializer)
        }

        fn read<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            <(usize, Owned<f64>)>::deserialize(deserializer)
                .map(|(index, Owned(value))| (index, value))
        }
    }
}

/// Encode the digest through its public centroid snapshot interface.
pub(crate) mod digest {
    use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
    use tdigest::{Centroid, TDigest};

    #[derive(Serialize, Deserialize)]
    struct CentroidState {
        mean: f64,
        weight: f64,
    }

    #[derive(Serialize, Deserialize)]
    struct DigestState {
        centroids: Vec<CentroidState>,
        max_size: usize,
        #[serde(with = "super::float")]
        sum: f64,
        count: f64,
        min: Option<f64>,
        max: Option<f64>,
    }

    pub fn serialize<S: Serializer>(digest: &TDigest, serializer: S) -> Result<S::Ok, S::Error> {
        DigestState {
            centroids: digest
                .centroids()
                .iter()
                .map(|c| CentroidState {
                    mean: c.mean(),
                    weight: c.weight(),
                })
                .collect(),
            max_size: digest.max_size(),
            sum: digest.sum(),
            count: digest.count(),
            min: digest.min(),
            max: digest.max(),
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<TDigest, D::Error> {
        let state = DigestState::deserialize(deserializer)?;
        let weight: f64 = state.centroids.iter().map(|c| c.weight).sum();
        let valid_extrema = match (state.min, state.max) {
            (Some(min), Some(max)) if !state.centroids.is_empty() => {
                min.is_finite() && max.is_finite() && min <= max
            }
            (None, None) => state.centroids.is_empty(),
            _ => false,
        };
        if state.max_size == 0
            || state.centroids.len() > state.max_size
            || !state.count.is_finite()
            || !weight.is_finite()
            || (state.count - weight).abs() > 1e-12 * weight
            || !valid_extrema
            || state
                .centroids
                .iter()
                .any(|c| !c.mean.is_finite() || !c.weight.is_finite() || c.weight <= 0.0)
        {
            return Err(D::Error::custom("invalid t-digest state"));
        }
        let centroids = state
            .centroids
            .iter()
            .map(|c| Centroid::new(c.mean, c.weight))
            .collect();
        Ok(TDigest::new(
            centroids,
            state.sum,
            state.count,
            state.max,
            state.min,
            state.max_size,
        ))
    }
}
