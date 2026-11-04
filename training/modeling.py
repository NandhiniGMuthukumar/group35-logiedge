"""The required MLP and float32 TFLite conversion, with lazy TensorFlow import."""

import os
from functools import lru_cache

import numpy as np

from training.evaluation import fold_first_kernel


@lru_cache(maxsize=1)
def get_tf():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def build_mlp():
    tf = get_tf()
    inputs = tf.keras.Input(shape=(6,), dtype=tf.float32, name="features")
    hidden = tf.keras.layers.Dense(32, activation="relu", name="hidden_1")(inputs)
    hidden = tf.keras.layers.Dense(16, activation="relu", name="hidden_2")(hidden)
    probabilities = tf.keras.layers.Dense(3, activation="softmax", name="probabilities")(hidden)
    model = tf.keras.Model(inputs, probabilities, name="logibridge_mlp")
    if model.count_params() != 803:
        raise ValueError("The 6-32-16-3 MLP should have 803 parameters")
    return model


def fold_conditioned_model(conditioned_model, scales):
    """Fold training-only x/scales into the first Dense kernel for deployment."""
    tf = get_tf()
    exported = tf.keras.models.clone_model(conditioned_model)
    weights = [np.array(value, copy=True) for value in conditioned_model.get_weights()]
    weights[0] = fold_first_kernel(weights[0], scales)
    exported.set_weights(weights)
    return exported


def convert_fp32(model):
    tf = get_tf()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def predict_tflite_fp32(model_content, features):
    tf = get_tf()
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("TFLite evaluation requires finite (N, 6) inputs")
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=1)
    interpreter.allocate_tensors()
    input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if (input_info["dtype"] != np.float32 or output_info["dtype"] != np.float32 or
            list(input_info["shape"]) != [1, 6] or list(output_info["shape"]) != [1, 3]):
        raise ValueError("Expected float32 TFLite input (1, 6) and output (1, 3)")
    predictions = []
    for row in values:
        interpreter.set_tensor(input_info["index"], row[None, :])
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_info["index"])[0].copy())
    return np.asarray(predictions, dtype=np.float32)
