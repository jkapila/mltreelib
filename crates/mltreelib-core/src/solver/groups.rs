//! Group metadata representations and validation for penalized and panel regression.
//!
//! Provides [`GroupSpec`] and [`GroupKind`] supporting:
//! - Coefficient groups for Group Lasso, Sparse-Group Lasso, and grouped monotone constraints
//! - Panel entity keys for within-transformation and entity fixed-effects
//! - Hierarchical groups for multi-level nested pooling

use std::collections::{HashMap, HashSet};
use std::fmt;

/// The nature of a group declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GroupKind {
    /// Named set of predictor columns. Group lasso selects or rejects these together.
    CoefficientGroup,
    /// Observation unit key for repeated measurements in panel models.
    PanelEntity,
    /// Nested entity level (e.g. store nested within department) for hierarchical pooling.
    HierarchicalGroup,
}

/// Specification of a single group.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupSpec {
    /// Kind of group (coefficient group, panel entity, or hierarchical level).
    pub kind: GroupKind,
    /// Unique identifier / label for the group.
    pub name: String,
    /// Member indices (column indices for `CoefficientGroup`, row indices for `PanelEntity`).
    pub members: Vec<usize>,
    /// Optional custom penalty multiplier. If `None`, defaults to $\sqrt{|\text{members}|}$.
    pub weight: Option<f64>,
}

impl GroupSpec {
    /// Creates a new `GroupSpec` with default penalty weight.
    pub fn new(kind: GroupKind, name: impl Into<String>, members: Vec<usize>) -> Self {
        Self {
            kind,
            name: name.into(),
            members,
            weight: None,
        }
    }

    /// Sets an explicit penalty weight multiplier for this group.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Effective penalty weight: explicit weight if set, otherwise $\sqrt{|\text{members}|}$.
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        self.weight.unwrap_or_else(|| (self.members.len() as f64).sqrt())
    }
}

/// Errors raised when validating group declarations.
#[derive(Clone, Debug, PartialEq)]
pub enum GroupValidationError {
    /// Group contains an index exceeding the allowable range.
    IndexOutOfRange {
        group: String,
        index: usize,
        max_allowed: usize,
    },
    /// Intercept column (column 0) was included in a coefficient group.
    InterceptIncluded { group: String },
    /// Two coefficient groups share the same predictor column index.
    OverlappingCoefficientGroups {
        column: usize,
        group1: String,
        group2: String,
    },
    /// A group declaration has no members.
    EmptyGroup { group: String },
    /// Non-positive or non-finite penalty weight.
    InvalidWeight { group: String, weight: f64 },
}

impl fmt::Display for GroupValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange {
                group,
                index,
                max_allowed,
            } => write!(
                f,
                "Group '{group}' member index {index} out of range (max allowed: {max_allowed})"
            ),
            Self::InterceptIncluded { group } => write!(
                f,
                "Intercept (column 0) cannot be a member of coefficient group '{group}'"
            ),
            Self::OverlappingCoefficientGroups {
                column,
                group1,
                group2,
            } => write!(
                f,
                "Overlapping coefficient groups detected: column {column} belongs to both '{group1}' and '{group2}'"
            ),
            Self::EmptyGroup { group } => {
                write!(f, "Group '{group}' cannot have empty members")
            }
            Self::InvalidWeight { group, weight } => {
                write!(
                    f,
                    "Group '{group}' has invalid penalty weight {weight}; must be positive and finite"
                )
            }
        }
    }
}

impl std::error::Error for GroupValidationError {}

/// Validates group declarations against design matrix dimensions.
///
/// Rules enforced:
/// - Every group must have at least one member.
/// - Member indices must be strictly `< n_features` (for `CoefficientGroup`) or `< n_rows` (for `PanelEntity`).
/// - The intercept (column 0 when `has_intercept` is true) cannot be a member of any coefficient group.
/// - `CoefficientGroup`s must be mutually disjoint (no overlapping columns).
/// - Weights, if specified, must be positive and finite.
pub fn validate_groups(
    groups: &[GroupSpec],
    n_features: usize,
    has_intercept: bool,
    n_rows: Option<usize>,
) -> Result<(), GroupValidationError> {
    let mut claimed_columns: HashMap<usize, &str> = HashMap::new();

    for group in groups {
        if group.members.is_empty() {
            return Err(GroupValidationError::EmptyGroup {
                group: group.name.clone(),
            });
        }

        if let Some(w) = group.weight {
            if !w.is_finite() || w <= 0.0 {
                return Err(GroupValidationError::InvalidWeight {
                    group: group.name.clone(),
                    weight: w,
                });
            }
        }

        match group.kind {
            GroupKind::CoefficientGroup => {
                for &idx in &group.members {
                    if has_intercept && idx == 0 {
                        return Err(GroupValidationError::InterceptIncluded {
                            group: group.name.clone(),
                        });
                    }
                    if idx >= n_features {
                        return Err(GroupValidationError::IndexOutOfRange {
                            group: group.name.clone(),
                            index: idx,
                            max_allowed: n_features.saturating_sub(1),
                        });
                    }
                    if let Some(&existing) = claimed_columns.get(&idx) {
                        return Err(GroupValidationError::OverlappingCoefficientGroups {
                            column: idx,
                            group1: existing.to_string(),
                            group2: group.name.clone(),
                        });
                    }
                    claimed_columns.insert(idx, &group.name);
                }
            }
            GroupKind::PanelEntity | GroupKind::HierarchicalGroup => {
                if let Some(rows) = n_rows {
                    for &idx in &group.members {
                        if idx >= rows {
                            return Err(GroupValidationError::IndexOutOfRange {
                                group: group.name.clone(),
                                index: idx,
                                max_allowed: rows.saturating_sub(1),
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Normalizes groups into deterministic canonical order.
///
/// Groups are sorted by name ascending, and members within each group are sorted ascending and deduplicated.
pub fn canonicalize_groups(groups: &mut [GroupSpec]) {
    groups.sort_by(|a, b| a.name.cmp(&b.name));
    for group in groups.iter_mut() {
        group.members.sort_unstable();
        group.members.dedup();
    }
}

/// Inherits group membership for missing-value indicator columns in `ColumnSliceView`.
///
/// In `ColumnSliceView`, active feature $k \in \{0 \dots nc-1\}$ resides at design column $1 + k$.
/// When indicator expansion is active (`with_indicators = true`), the missing indicator
/// for feature $k$ resides at design column $1 + nc + k$.
///
/// This helper expands coefficient group members such that if column $1 + k$ is in a group,
/// column $1 + nc + k$ is automatically appended to that group.
pub fn expand_missing_indicators(groups: &[GroupSpec], n_chosen: usize) -> Vec<GroupSpec> {
    let mut out = groups.to_vec();
    for group in out.iter_mut() {
        if group.kind != GroupKind::CoefficientGroup {
            continue;
        }
        let mut new_members = HashSet::new();
        for &m in &group.members {
            new_members.insert(m);
            // If m corresponds to a chosen feature: 1 <= m <= n_chosen
            if m >= 1 && m <= n_chosen {
                let indicator_col = m + n_chosen;
                new_members.insert(indicator_col);
            }
        }
        let mut sorted: Vec<usize> = new_members.into_iter().collect();
        sorted.sort_unstable();
        group.members = sorted;
    }
    canonicalize_groups(&mut out);
    out
}
