# 07 — mlxtend suite (classical models + meta-estimators)

**Source:** https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.classifier/
and `.../mlxtend.regressor/` (v0.24.0). Members verified from the pages.

## mlxtend.classifier members

`Adaline`, `Perceptron`, `LogisticRegression`, `SoftmaxRegression`,
`MultiLayerPerceptron`, `OneRClassifier` (see 01), `EnsembleVoteClassifier`
(see 04), `StackingClassifier`, `StackingCVClassifier` (see 04).

## Adaline (ADAptive LInear NEuron)

**Idea:** linear threshold unit trained on squared error of the **linear
output** (before the step function), via normal equations or GD/SGD.

```text
init w = 0
for epoch in 1..E:
    for minibatch:
        net = X @ w + b
        grad = X.T @ (net - y) / n
        w -= eta * grad;  b -= eta * mean(net - y)
predict = step(net)        # binary {0, 1}
```

**Knobs:** eta, epochs, minibatches (None = normal equations / 1 = GD / n =
SGD). **Note:** the SSE-trained perceptron — reference point for linear
classifiers.

## Perceptron

**Idea:** update only on mistakes: `w += eta * (y - pred) * x`.

```text
for epoch in 1..E, for (x_i, y_i) in shuffled data:
    pred = step(x_i @ w + b)
    if pred != y_i: w += eta * (y_i - pred) * x_i; b += eta * (y_i - pred)
```

**Knobs:** eta, epochs, random_seed. **Note:** converges iff linearly
separable — a good sanity baseline.

## LogisticRegression (binary) and SoftmaxRegression (multiclass)

**Idea:** linear scores through the sigmoid/softmax; minimize log loss by
GD/SGD (or normal equations where applicable); optional L2.

```text
for epoch in 1..E:
    p = sigmoid(X @ w + b)             # softmax for multiclass
    grad = X.T @ (p - y) / n + l2 * w
    w -= eta * grad
predict = p >= 0.5 (or argmax of softmax)
```

**Knobs:** eta, epochs, minibatches, l2 regularization.

## MultiLayerPerceptron

**Idea:** one-hidden-layer perceptron with sigmoid hidden units, softmax or
sigmoid output, backprop with optional L2 and momentum.

```text
for epoch in 1..E, for minibatch:
    h = sigmoid(X @ W1 + b1)
    out = softmax/sigmoid(h @ W2 + b2)
    backprop gradients; update W1, W2 with eta (+ momentum, l2)
```

**Knobs:** hidden_layers sizes, eta, epochs, momentum, l2, minibatches.

## mlxtend.regressor members

`LinearRegression` (OLS via direct/QR/SVD or SGD),
`StackingRegressor`, `StackingCVRegressor` (see 04 for the stacking
pseudocode).

### LinearRegression

```text
direct/QR/SVD: w = argmin ||Xw - y||^2 via normal equations / QR / SVD
sgd: w -= eta * X.T @ (Xw - y) / n per minibatch
predict = X @ w + b
```

**Knobs:** method = direct | qr | svd | sgd, eta/epochs/minibatches for sgd.
**Note:** our Rust leaf models will do QR via faer — same recipe as
modeltrees' `dqrls` port, without reinventing the factorization.

## Where mlxtend sits in our program

mlxtend = the **classical layer + meta-ensemble toolbox** (stacking/blending/
voting) built on sklearn. We do not depend on it; we reimplement the recipes
natively (sklearn-free rule) and keep these notes as the reference for
semantics (e.g. StackingCV's out-of-fold protocol in the mixing phase).
