import json
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "mnist_train.csv"
TEST_FILE = BASE_DIR / "mnist_test.csv"
OUTPUT_FILE = BASE_DIR / "model-weights.js"


def log(msg):
    print(f"[INFO] {msg}")


def load_train_file(path: Path):
    log(f"Loading train file: {path}")
    df = pd.read_csv(path, header=None)

    if df.shape[1] != 785:
        raise ValueError(
            f"Train file must have 785 columns (1 label + 784 pixels). "
            f"Found {df.shape[1]} columns."
        )

    y = df.iloc[:, 0].astype("int64").to_numpy()
    X = df.iloc[:, 1:].astype("float32").to_numpy()

    return X, y


def load_test_file(path: Path):
    log(f"Loading test file: {path}")
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 785:
        y = df.iloc[:, 0].astype("int64").to_numpy()
        X = df.iloc[:, 1:].astype("float32").to_numpy()
        has_labels = True

    elif df.shape[1] == 784:
        y = None
        X = df.astype("float32").to_numpy()
        has_labels = False

    else:
        raise ValueError(
            f"Test file must have either 785 columns (label + pixels) "
            f"or 784 columns (pixels only). Found {df.shape[1]} columns."
        )

    return X, y, has_labels


def normalize_pixels(X):
    return X / 255.0


def avg_pool_2x2(X):
    """
    Convert (n, 784) -> (n, 196)
    using 2x2 average pooling from 28x28 to 14x14
    """
    n = X.shape[0]
    X_reshaped = X.reshape(n, 28, 28)

    pooled = (
        X_reshaped[:, 0::2, 0::2] +
        X_reshaped[:, 0::2, 1::2] +
        X_reshaped[:, 1::2, 0::2] +
        X_reshaped[:, 1::2, 1::2]
    ) / 4.0

    return pooled.reshape(n, 196)


def train_model(X_train, y_train):
    log("Training MLP model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=20,
        verbose=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, name="dataset"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    log(f"Accuracy on {name}: {acc:.6f}")
    return acc


def build_weights_hidden(model):
    """
    Build JS-compatible hidden weights:
    196 inputs + 1 bias for each of 100 neurons
    => 19700 values
    """
    W = model.coefs_[0]       # shape (196, 100)
    b = model.intercepts_[0]  # shape (100,)

    weights = []
    for hidden_idx in range(W.shape[1]):
        for input_idx in range(W.shape[0]):
            weights.append(float(W[input_idx, hidden_idx]))
        weights.append(float(b[hidden_idx]))

    return weights


def build_weights_output(model):
    """
    Build JS-compatible output weights:
    100 hidden activations + 1 bias for each of 10 outputs
    => 1010 values
    """
    W = model.coefs_[1]       # shape (100, 10)
    b = model.intercepts_[1]  # shape (10,)

    weights = []
    for output_idx in range(W.shape[1]):
        for hidden_idx in range(W.shape[0]):
            weights.append(float(W[hidden_idx, output_idx]))
        weights.append(float(b[output_idx]))

    return weights


def write_model_weights(weights_hidden, weights_output, output_file: Path):
    log(f"Writing JS weights file: {output_file}")

    hidden_json = json.dumps(weights_hidden, ensure_ascii=False)
    output_json = json.dumps(weights_output, ensure_ascii=False)

    content = f"""window.MNIST_WEIGHTS = {{
  weightsHidden: {hidden_json},
  weightsOutput: {output_json}
}};
"""
    output_file.write_text(content, encoding="utf-8")


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    X_train_raw, y_train = load_train_file(TRAIN_FILE)
    X_test_raw, y_test, test_has_labels = load_test_file(TEST_FILE)

    log(f"Train shape: X={X_train_raw.shape}, y={y_train.shape}")
    log(f"Test shape: X={X_test_raw.shape}, y={'available' if test_has_labels else 'not available'}")

    X_train = normalize_pixels(X_train_raw)
    X_test = normalize_pixels(X_test_raw)

    log("Applying 2x2 average pooling...")
    X_train_pool = avg_pool_2x2(X_train)
    X_test_pool = avg_pool_2x2(X_test)

    log(f"Pooled train shape: {X_train_pool.shape}")
    log(f"Pooled test shape: {X_test_pool.shape}")

    model = train_model(X_train_pool, y_train)

    evaluate_model(model, X_train_pool, y_train, "train")

    if test_has_labels:
        evaluate_model(model, X_test_pool, y_test, "test")
    else:
        log("Test file has no labels; skipping test accuracy.")

    weights_hidden = build_weights_hidden(model)
    weights_output = build_weights_output(model)

    log(f"weightsHidden length = {len(weights_hidden)} (expected 19700)")
    log(f"weightsOutput length = {len(weights_output)} (expected 1010)")

    if len(weights_hidden) != 19700:
        raise ValueError(f"Unexpected weightsHidden length: {len(weights_hidden)}")

    if len(weights_output) != 1010:
        raise ValueError(f"Unexpected weightsOutput length: {len(weights_output)}")

    write_model_weights(weights_hidden, weights_output, OUTPUT_FILE)

    log("Done. model-weights.js has been created successfully.")


if __name__ == "__main__":
    main()