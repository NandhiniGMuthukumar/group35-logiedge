"""Same TFLite built-in CPU backend used in the measured F2 comparison."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data_pipeline.normalization import FrozenNormalizer,file_sha256,metadata_path
from data_pipeline.preprocessing import FEATURE_NAMES,RECIPE

LABELS=("Normal","Warning","Critical")


class LocalModel:
    def __init__(self,model_path,stats_path,registry_path):
        self.path=Path(model_path)
        self.registry=json.loads(Path(registry_path).read_text())
        self.normalizer=FrozenNormalizer.load(stats_path)
        if (self.registry.get("feature_names")!=list(FEATURE_NAMES) or self.registry.get("recipe")!=RECIPE or
                self.registry.get("normalization_stats_sha256")!=self.normalizer.stats_sha256 or
                self.registry.get("normalization_metadata_sha256")!=file_sha256(metadata_path(stats_path))):
            raise ValueError("Runtime registry and frozen normalization do not match")
        content=self.path.read_bytes();self.sha256=hashlib.sha256(content).hexdigest()
        matches=[entry for entry in self.registry["models"] if entry["sha256"]==self.sha256]
        if len(matches)!=1:raise ValueError("MODEL_PATH is not a verified model in the runtime registry")
        self.variant=matches[0]["variant"]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
        import tensorflow as tf
        if tf.__version__!="2.14.1":raise ValueError("This CPU runtime requires the benchmarked TensorFlow 2.14.1")
        self.interpreter=tf.lite.Interpreter(model_content=content,num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.interpreter.allocate_tensors()
        self.input=self.interpreter.get_input_details()[0];self.output=self.interpreter.get_output_details()[0]
        expected=np.float32 if self.variant=="M1_FP32" else np.int8
        if (self.input["dtype"]!=expected or self.output["dtype"]!=expected or
                list(self.input["shape"])!=[1,6] or list(self.output["shape"])!=[1,3]):
            raise ValueError("Unexpected model tensor signature")
        self.int8=expected==np.int8

    def predict(self,features):
        normalized=self.normalizer.transform(features)
        if normalized.shape!=(6,):raise ValueError("One six-feature vector is required")
        clipping=0
        if self.int8:
            scale,zero=self.input["quantization"]
            rounded=np.rint(normalized.astype(np.float64)/scale+zero)
            clipping=int(((rounded < -128)|(rounded > 127)).sum())
            encoded=np.clip(rounded,-128,127).astype(np.int8)
        else:encoded=normalized
        self.interpreter.set_tensor(self.input["index"],encoded[None,:]);self.interpreter.invoke()
        output=self.interpreter.get_tensor(self.output["index"])[0]
        if self.int8:
            scale,zero=self.output["quantization"]
            probabilities=((output.astype(np.float64)-zero)*scale).astype(np.float32)
        else:probabilities=output
        if not np.all(np.isfinite(probabilities)):raise ValueError("Nonfinite model output")
        return {"model_class":int(output.argmax()),"probabilities":probabilities.astype(float).tolist(),
                "normalized_features":normalized.astype(float).tolist(),"clipped_feature_count":clipping}
