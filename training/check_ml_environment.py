"""Check the pinned ML environment before using any project training data."""

import importlib.metadata
import platform

import numpy as np

from training.modeling import build_mlp, get_tf

EXPECTED = {
    "tensorflow-cpu": "2.14.1", "keras": "2.14.0",
    "tensorflow-model-optimization": "0.8.0",
    "numpy": "1.26.4", "paho-mqtt": "2.1.0", "psutil": "6.1.1",
    "matplotlib": "3.8.4",
}


def check_environment():
    tf = get_tf()
    import tensorflow_model_optimization as tfmot

    if tuple(map(int, platform.python_version_tuple()[:2])) != (3, 11):
        raise ValueError("Use the project's Python 3.11 ML container")
    print("Python:", platform.python_version())
    for package, expected in EXPECTED.items():
        observed = importlib.metadata.version(package)
        print(f"{package}: {observed}")
        if observed != expected:
            raise ValueError(f"{package}: expected {expected}, observed {observed}")
    # Only check layer-type compatibility. This is not the M3 pruning experiment.
    wrapped = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(2))
    if wrapped is None:
        raise ValueError("Pruning wrapper could not be constructed")
    model = build_mlp()
    output = model(np.zeros((1, 6), dtype=np.float32), training=False).numpy()
    if output.shape != (1, 3) or not np.allclose(output.sum(axis=1), 1, atol=1e-5):
        raise ValueError("MLP output shape or probabilities are invalid")
    print("MLP architecture: 6-32-16-3 | parameters:", model.count_params())
    print("Visible GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Keras/pruning API compatibility: checked")
    print("Model training has not been performed by this environment check")


if __name__ == "__main__":
    check_environment()
