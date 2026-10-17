# Release Notes — MLP in JAX from Scratch v1.0.0

An end-to-end demo of a multilayer perceptron built with pure JAX — no Flax, no Optax, no Keras. Every component (parameter init, forward pass, loss, gradients, SGD update loop) is written by hand on top of `jax` and `jax.numpy`.

## ✨ What's in this release

- **Pure-JAX MLP implementation** (`mlpBuild.py`)
  - Forward pass: dense layers (`x @ W + b`) with ReLU activations, linear logits at the output
  - Numerically stable cross-entropy loss via `log_softmax`
  - Gradients computed with `jax.grad` — no manual derivations
  - Hand-rolled full-batch SGD update loop
- **Synthetic dataset generation**
  - Standard-normal feature batches with argmax-based class labels
  - One-hot label encoding
- **Training demo** (`python mlpBuild.py`)
  - Configurable seed, batch size, layer sizes, learning rate, epochs
  - Reports loss/accuracy before and after training and compares predictions to labels

## 🚀 Getting started

```bash
pip install -U "jax[cpu]"
python mlpBuild.py
```

Example output (seed 0, 200 epochs):

```
Initial loss:     1.3831
Initial accuracy: 0.3125
Final loss:       0.1521
Final accuracy:   1.0000
```

Note: the layer-shape printout is currently buggy in this build (shows scalar shapes) — cosmetic only, training works fine.

## 📐 Default config

| Hyperparameter | Value |
|---|---|
| Architecture | 8 → 16 (ReLU) → 16 (ReLU) → 4 |
| Batch size | 32 |
| Learning rate | 0.1 |
| Epochs | 200 |

## 📋 Notes

- Runs on CPU — the model is tiny; no GPU/TPU required.
- Requires Python 3.x and JAX ≥ 0.4.
- Labels are derived from the input features themselves (as instructed by the exercise), so the task is fully learnable and accuracy reaches 100% on the training batch.

## 🙏 Thanks

Thanks to everyone trying this out — feedback and PRs welcome!
