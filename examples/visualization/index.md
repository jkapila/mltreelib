# Visualization

Real-work visualization: tree diagrams and diagnostic plots. The division of labor is deliberate — **graphviz draws the tree** (we only emit DOT; layout is a solved problem we don't reimplement) and **matplotlib draws the plots**.

Script: [`examples/visualization.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/visualization.py>).

## Tree diagrams via graphviz

```
import pygraphviz as pgv

dot = model.to_dot()                    # DOT description, heap-numbered nodes
g = pgv.AGraph(string=dot)
g.layout(prog="dot")                    # graphviz does the layout
g.draw("docs/assets/diamonds_tree.svg")
```

`to_dot()` emits one box per node (split condition, `n`, deviance / leaf `yval`), edges labeled with `< t` / `>= t`, node ids matching the rpart text print — so the printed tree and the diagram line up node-for-node.

Here is the fitted diamonds tree (depth 6, 200-row leaves), rendered by graphviz from `to_dot()` output:

## Diagnostic plots

Predicted-vs-actual and residual panels (matplotlib, guarded so the library itself never depends on it):

The tree tracks price almost linearly at the top end but fans out at high prices — the classic heteroscedastic diamond signal.

## Run it

```
uv run --with pandas --with pygraphviz --with matplotlib python examples/visualization.py
```

```
tree diagram: docs/assets/diamonds_tree.svg (42 leaves, rendered by graphviz)
diagnostics: docs/assets/diamonds_diagnostics.png
test rmse: 573.6
```

No pygraphviz? The DOT file is still there (`to_dot()` output / `dot -Tsvg file.dot -o tree.svg` with the graphviz binary). The engine is always graphviz either way.

Next: [Benchmarks — ELO](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) for what the model is worth at scale, or [Inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) for the numbers behind the pictures.
