//! Context Anomaly Tree (Suffix Context Anomaly Detector).
//!
//! Based on Context Tree Weighting (Willems 1995) and Probabilistic Suffix Trees (Ron 1996):
//! Evaluates sequence transitions and hierarchical categorical configurations.
//! Each node stores conditional transition frequencies given a context prefix.
//! Anomaly score is the negative log-likelihood (information surprise in bits):
//!   Surprise(s | context) = -log2 P_KT(s | context)
//! Traversal provides instant contextual explanations for rare or impossible transitions.

use std::collections::HashMap;

/// A node in the Context Anomaly Tree.
#[derive(Clone, Debug, Default)]
pub struct ContextNode {
    /// Symbol frequency counts conditioned on this context prefix.
    pub symbol_counts: HashMap<u64, usize>,
    /// Total count of observations passing through this context.
    pub total_count: usize,
    /// Child transitions: symbol -> child node index.
    pub children: HashMap<u64, usize>,
}

/// Suffix Context Anomaly Tree estimator.
#[derive(Clone, Debug)]
pub struct ContextAnomalyTree {
    pub max_depth: usize,
    pub alphabet_size: usize,
    pub nodes: Vec<ContextNode>,
}

impl Default for ContextAnomalyTree {
    fn default() -> Self {
        Self {
            max_depth: 5,
            alphabet_size: 10,
            nodes: vec![ContextNode::default()],
        }
    }
}

impl ContextAnomalyTree {
    pub fn new(max_depth: usize, alphabet_size: usize) -> Self {
        Self {
            max_depth,
            alphabet_size: alphabet_size.max(2),
            nodes: vec![ContextNode::default()],
        }
    }

    /// Insert a categorical sequence / path: (context_prefix, next_symbol).
    pub fn insert(&mut self, context: &[u64], symbol: u64) {
        let mut curr = 0;
        self.nodes[curr].total_count += 1;
        *self.nodes[curr].symbol_counts.entry(symbol).or_insert(0) += 1;

        let depth_limit = context.len().min(self.max_depth);
        for &ctx_sym in &context[..depth_limit] {
            let next_idx = if let Some(&child) = self.nodes[curr].children.get(&ctx_sym) {
                child
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(ContextNode::default());
                self.nodes[curr].children.insert(ctx_sym, new_idx);
                new_idx
            };

            curr = next_idx;
            self.nodes[curr].total_count += 1;
            *self.nodes[curr].symbol_counts.entry(symbol).or_insert(0) += 1;
        }
    }

    /// Fit the context tree on a collection of categorical sequences / feature paths.
    pub fn fit(&mut self, sequences: &[Vec<u64>]) {
        self.nodes = vec![ContextNode::default()];
        for seq in sequences {
            for i in 1..seq.len() {
                let ctx_start = if i > self.max_depth { i - self.max_depth } else { 0 };
                let context = &seq[ctx_start..i];
                let symbol = seq[i];
                self.insert(context, symbol);
            }
        }
    }

    /// Compute Krichevsky-Trofimov (KT) smoothed conditional probability:
    ///   P_KT(s | context) = (count(s) + 0.5) / (total + alphabet_size / 2)
    pub fn conditional_prob(&self, context: &[u64], symbol: u64) -> f64 {
        let mut curr = 0;
        let depth_limit = context.len().min(self.max_depth);

        for &ctx_sym in &context[..depth_limit] {
            if let Some(&child) = self.nodes[curr].children.get(&ctx_sym) {
                curr = child;
            } else {
                break;
            }
        }

        let node = &self.nodes[curr];
        let sym_cnt = node.symbol_counts.get(&symbol).copied().unwrap_or(0) as f64;
        let tot_cnt = node.total_count as f64;
        let alph = self.alphabet_size as f64;

        (sym_cnt + 0.5) / (tot_cnt + 0.5 * alph)
    }

    /// Compute surprise anomaly score in bits: -log2 P_KT(symbol | context).
    /// High surprise indicates an anomalous symbol given its context history.
    pub fn surprise_score(&self, context: &[u64], symbol: u64) -> f64 {
        let p = self.conditional_prob(context, symbol);
        -p.log2()
    }
}
