//! Rendering: the rpart-pinned text print and the Graphviz DOT export.
//! Number formatting follows R's `format(signif(x, 7))` conventions.

use crate::model_tree::ModelTree;
/// Decimals needed to show `x` with up to 7 significant digits, trailing
/// zeros trimmed (R trims the column's common format to what values need).
fn decs_for(x: f64) -> usize {
    if x == 0.0 {
        return 0;
    }
    let d = (7.0 - 1.0 - x.abs().log10().floor()).clamp(0.0, 20.0) as usize;
    let s = format!("{:.*}", d, x);
    let trimmed = s.trim_end_matches('0');
    let nd = trimmed.split_once('.').map(|(_, frac)| frac.len()).unwrap_or(0);
    nd.min(d)
}

/// R's `format(signif(x, 7))` over a whole COLUMN: one common number of
/// decimals (fixed), or all-scientific `d.ddddddde+NN` when any nonzero
/// magnitude is < 1e-3 or any is >= 1e7. Pinned from rpart's print output.
fn fmt_column(vals: &[f64]) -> Vec<String> {
    let sci = vals
        .iter()
        .any(|v| (*v != 0.0 && v.abs() < 1e-3) || v.abs() >= 1e7);
    if sci {
        vals.iter()
            .map(|v| {
                let s = format!("{:.6e}", v);
                let (m, e) = s.split_once('e').unwrap();
                format!("{}e{:+03}", m, e.parse::<i32>().unwrap())
            })
            .collect()
    } else {
        let d = vals.iter().map(|v| decs_for(*v)).max().unwrap_or(0);
        vals.iter().map(|v| format!("{:.*}", d, v)).collect()
    }
}

/// rpart's `formatg`: 7 significant digits, trailing zeros trimmed —
/// the split cutpoint label.
fn fmt_cut(x: f64) -> String {
    let s = format!("{:.*}", decs_for(x), x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}
impl ModelTree {
    /// Graphviz DOT for the tree. We only describe the graph - layout and
    /// rendering belong to graphviz (the wheel we do not reinvent):
    /// `pgv.AGraph(string=model.to_dot()).draw("tree.svg", prog="dot")`.
    pub fn to_dot(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let mut s = String::from(
            "digraph tree {\n  rankdir=LR;\n  node [shape=box, style=\"rounded,filled\", fillcolor=\"#fafafa\", fontname=\"monospace\", fontsize=10];\n  edge [fontname=\"monospace\", fontsize=9, color=\"#888888\"];\n",
        );
        // Heap numbering (root 1, children 2i/2i+1) - same ids as the text
        // print, so the two outputs line up.
        let mut num = vec![0usize; self.nodes.len()];
        fn number(t: &ModelTree, i: usize, k: usize, num: &mut Vec<usize>) {
            num[i] = k;
            if let (Some(_), Some(l), Some(r)) = (t.nodes[i].split, t.nodes[i].left, t.nodes[i].right) {
                number(t, l, k * 2, num);
                number(t, r, k * 2 + 1, num);
            }
        }
        number(self, self.root, 1, &mut num);
        for (i, nd) in self.nodes.iter().enumerate() {
            let (l1, l2) = match nd.split {
                Some(sp) => (
                    match sp.cat {
                        Some(ci) => format!(
                            "{} in {{{}}}",
                            self.col_names[sp.col],
                            self.cat_parts[ci]
                                .iter()
                                .map(|v| fmt_cut(*v))
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                        None => format!("{} < {}", self.col_names[sp.col], fmt_cut(sp.threshold)),
                    },
                    format!("n={}  dev={:.3}", nd.stats.n, nd.stats.sse),
                ),
                None => (
                    "leaf".to_string(),
                    format!("n={}  yval={:.3}", nd.stats.n, nd.stats.mean),
                ),
            };
            let stroke = if nd.split.is_some() { "#333333" } else { "#22aa77" };
            s += &format!(
                "  n{} [label=\"{}\\n{}\", color=\"{stroke}\"];\n",
                num[i],
                esc(&l1),
                esc(&l2)
            );
        }
        for (i, nd) in self.nodes.iter().enumerate() {
            if let (Some(sp), Some(l), Some(r)) = (nd.split, nd.left, nd.right) {
                let (el, er) = match sp.cat {
                    Some(ci) => {
                        let set = self.cat_parts[ci]
                            .iter()
                            .map(|v| fmt_cut(*v))
                            .collect::<Vec<_>>()
                            .join(",");
                        (format!("in {{{set}}}"), format!("not in {{{set}}}"))
                    }
                    None => {
                        let cut = fmt_cut(sp.threshold);
                        (format!("< {cut}"), format!(">= {cut}"))
                    }
                };
                s += &format!("  n{} -> n{} [label=\"{}\"];\n", num[i], num[l], esc(&el));
                s += &format!("  n{} -> n{} [label=\"{}\"];\n", num[i], num[r], esc(&er));
            }
        }
        s += "}\n";
        s
    }
}

/// rpart-style text print (grounded byte-for-byte on the installed
/// `rpart`'s output; see `references/repos/rpart/R/print.rpart.R`):
///
/// ```text
/// n= 20
///
/// node), split, n, deviance, yval
///       * denotes terminal node
///
///  1) root 20 6.900288e+01 2.5190520
///    2) x< 0.5 10 2.254216e+00 0.7237466
/// ```
///
/// Rows are preorder with binary-heap numbering (`format(node)` is
/// right-aligned to the widest number); indent is two spaces per depth;
/// terminal nodes are marked `*`.
impl std::fmt::Display for ModelTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Row {
            depth: usize,
            node: usize,
            label: String,
            n: usize,
            dev: f64,
            yval: f64,
            leaf: bool,
        }
        fn walk(
            t: &ModelTree,
            i: usize,
            num: usize,
            depth: usize,
            label: &str,
            out: &mut Vec<Row>,
        ) {
            let nd = &t.nodes[i];
            out.push(Row {
                depth,
                node: num,
                label: label.to_string(),
                n: nd.stats.n,
                dev: nd.stats.sse,
                yval: nd.leaf.as_ref().and_then(|l| l.center).unwrap_or(nd.stats.mean),
                leaf: nd.split.is_none(),
            });
            if let (Some(sp), Some(l), Some(r)) = (nd.split, nd.left, nd.right) {
                let name = &t.col_names[sp.col];
                let (ll, rl) = match sp.cat {
                    Some(ci) => {
                        let set = t.cat_parts[ci]
                            .iter()
                            .map(|v| fmt_cut(*v))
                            .collect::<Vec<_>>()
                            .join(",");
                        (format!("{name} in {{{set}}}"), format!("{name} not-in {{{set}}}"))
                    }
                    None => {
                        let cut = fmt_cut(sp.threshold);
                        (format!("{name}< {cut}"), format!("{name}>={cut}"))
                    }
                };
                walk(t, l, num * 2, depth + 1, &ll, out);
                walk(t, r, num * 2 + 1, depth + 1, &rl, out);
            }
        }
        let mut rows = Vec::new();
        walk(self, self.root, 1, 0, "root", &mut rows);
        let width = rows.iter().map(|r| r.node).max().unwrap_or(1).to_string().len();
        let devs = fmt_column(&rows.iter().map(|r| r.dev).collect::<Vec<_>>());
        let yvals = fmt_column(&rows.iter().map(|r| r.yval).collect::<Vec<_>>());

        writeln!(f, "n= {} ", rows[0].n)?;
        writeln!(f)?;
        writeln!(f, "node), split, n, deviance, yval")?;
        writeln!(f, "      * denotes terminal node")?;
        writeln!(f)?;
        for (k, r) in rows.iter().enumerate() {
            let term = if r.leaf { "*" } else { " " };
            writeln!(
                f,
                "{}{:>w$}) {} {} {} {} {}",
                "  ".repeat(r.depth),
                r.node,
                r.label,
                r.n,
                devs[k],
                yvals[k],
                term,
                w = width
            )?;
        }
        Ok(())
    }
}
