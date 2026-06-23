# Neural Network Sandbox — Interactive MNIST Digit Classifier

> A browser-based, dependency-free Multilayer Perceptron (MLP) that recognizes handwritten digits in real time. Draw a digit, watch the forward pass animate layer by layer, and see the Softmax probability distribution update live — all powered by vanilla JavaScript with no TensorFlow.js, no ONNX, no model files.

**Live demo:**

- 🌐 English version: [ann-mnist.mohsenfathipour.com](https://ann-mnist.mohsenfathipour.com/)
- 🇮🇷 Farsi version: [ann-mnist-farsi.mohsenfathipour.com](https://ann-mnist-farsi.mohsenfathipour.com/)

![License: MIT](https://img.shields.io/badge/License-MIT-6366f1.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-f7df1e.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-10b981.svg)

---

## What Is This?

This project is an **interactive neural network visualizer** built for learning and teaching purposes. It trains a small MLP on the MNIST dataset using Python, exports the learned weights as a plain JavaScript object, and then runs the entire forward pass inside the browser — no server, no ML runtime.

Key ideas demonstrated:

- How pixel values become a flattened input vector
- How average pooling reduces dimensionality (28×28 → 14×14)
- How ReLU activations fire (or don't) in the hidden layer
- How Softmax turns raw scores into a probability distribution
- How positive and negative weights shape what the network "looks for"

---

## Features

- **Live drawing canvas** — 28×28 pixel grid, mouse and touch supported
- **Animated forward pass** — GSAP-powered wave animation shows signal propagating Input → Hidden → Output
- **Weight visualization** — purple lines = positive weights, green lines = negative weights
- **Softmax bar chart** — real-time probability bars for all 10 digit classes
- **Network stats** — active neuron count, max activation, top confidence
- **Preprocessing pipeline** — shows centering, bounding-box crop, and 2×2 average pooling steps
- **No dependencies at runtime** — just HTML + CSS + JS; weights are hard-coded as a JS object

---

## Model Architecture

```
Input (28×28 px)
    ↓  center & crop to 20×20
    ↓  avg-pool 2×2
Flattened vector: 196 features + 1 bias node
    ↓  fully connected
Hidden layer: 100 neurons — ReLU activation  f(x) = max(0, x)
    ↓  fully connected
Output layer: 10 classes — Softmax activation  σ(zᵢ) = eᶻⁱ / Σeᶻʲ
```

| Layer        | Size             | Parameters             |
| ------------ | ---------------- | ---------------------- |
| Pooled input | 196 + 1 bias     | —                      |
| Hidden       | 100 neurons      | 197 × 100 = **19,700** |
| Output       | 10 classes (0–9) | 101 × 10 = **1,010**   |
| **Total**    |                  | **20,710 parameters**  |

Training results on MNIST:

| Split | Accuracy |
| ----- | -------- |
| Train | ~98.3%   |
| Test  | ~97.6%   |

---

## Project Structure

```
ANN/
├── train_from_csv.py   ← trains the MLP and exports weights to JS
├── model-weights.js    ← hard-coded weights (output of Python script)
├── index.html          ← UI layout and styles
├── app.js              ← forward pass, SVG renderer, GSAP animations
├── mnist_train.csv     ← MNIST training split (CSV format)
└── mnist_test.csv      ← MNIST test split (CSV format)
```

---

## Getting Started

### 1. Install Python dependencies

```bash
pip install numpy pandas scikit-learn
```

### 2. Unzip the dataset

Before training, extract the compressed CSV files:

```bash
tar -xzf mnist_train.csv.tar.gz
tar -xzf mnist_test.csv.tar.gz
```

This will extract `mnist_train.csv` and `mnist_test.csv` in the project directory.

### 3. Train the model and export weights

```bash
python train_from_csv.py
```

Expected output:

```
[INFO] Accuracy on train: 0.9832
[INFO] Accuracy on test:  0.9762
[INFO] Done. model-weights.js has been created successfully.
```

This writes `model-weights.js` containing the trained weights as a global `window.MNIST_WEIGHTS` object.

### 4. Open in a browser

Option A — serve locally (recommended):

```bash
python -m http.server 8000
# then open http://localhost:8000
```

Option B — open `index.html` directly in the browser (works in most cases, no server needed).

---

## CSV Data Format

Both `mnist_train.csv` and `mnist_test.csv` must follow the standard MNIST-CSV layout:

```csv
label,pixel0,pixel1,...,pixel783
5,0,0,0,...,128
7,0,0,0,...,255
```

- **label** — digit class (0–9)
- **pixel0 … pixel783** — grayscale pixel values (0–255), row-major order

If `label` is present in the test file, the script also reports test accuracy.

---

## Troubleshooting

| Error                               | Cause                               | Fix                                                                   |
| ----------------------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `ModuleNotFoundError`               | Missing Python package              | `pip install numpy pandas scikit-learn`                               |
| `Train/Test file not found`         | CSV files not in project folder     | Move `mnist_train.csv` and `mnist_test.csv` next to the script        |
| `window.MNIST_WEIGHTS is undefined` | `model-weights.js` missing or empty | Re-run `python train_from_csv.py`                                     |
| Blank canvas / no animation         | Browser blocked local file access   | Use `python -m http.server 8000` instead of opening the file directly |

---

## Limitations

This is a lightweight MLP, not a CNN. It is therefore more sensitive to drawing style differences compared to the MNIST training distribution (stroke thickness, centering, slant). For production digit recognition, a convolutional architecture would be more robust. For **learning, teaching, and R&D demos**, this project is the right tool.

---

## Why No TensorFlow.js?

The goal was to make the math as transparent as possible. By exporting raw weight arrays and implementing the forward pass manually (matrix multiply → ReLU → matrix multiply → Softmax), every computation is readable in plain JavaScript. There is no abstraction layer between the weights and the visualization.

---

## License

MIT © [Mohsen Fathipour](https://mohsenfathipour.com)
