"""Runs a TFLite model with LiteRT (the current TensorFlow Lite runtime; pip install ai-edge-litert).

A model is either a single .tflite file or a folder with a .tflite file and a manifest.json. Models with integer (int8) input and/or output
are handled: the picture is quantized with the model's own scale and zero point, and the result is turned back into floats.
"""
import json
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as e:
    raise ImportError("TFLite models need LiteRT: pip install ai-edge-litert") from e


class Pipeline:
    def __init__(self, path, threads=4):
        path = Path(path)
        if path.is_file():
            files = [path]
        else:                                          # a folder: one or more stage files listed in manifest.json (stage k feeds stage k+1)
            files = [path / name for name in json.loads((path / "manifest.json").read_text())["stages"]]
        self.stages = []
        for f in files:
            it = Interpreter(model_path=str(f), num_threads=threads)
            it.allocate_tensors()
            self.stages.append((it, it.get_input_details(), it.get_output_details()))

    def __call__(self, x):
        """x: float32 NCHW picture (0-1), or an integer NHWC picture already quantized for an integer-input model."""
        feeds = [np.ascontiguousarray(x.transpose(0, 2, 3, 1), dtype=np.float32) if x.dtype == np.float32 else x]        # NCHW -> NHWC
        for it, ins, outs in self.stages:
            for d in ins:
                v = next(f for f in feeds if tuple(f.shape) == tuple(d["shape"]))
                if d["dtype"] != np.float32 and v.dtype == np.float32:
                    scale, zero = d["quantization"]
                    info = np.iinfo(d["dtype"])
                    v = np.clip(np.round(v / scale) + zero, info.min, info.max).astype(d["dtype"])
                it.set_tensor(d["index"], v)
            it.invoke()
            feeds = [it.get_tensor(d["index"]) for d in outs]
            for j, d in enumerate(outs):
                if d["dtype"] != np.float32:
                    scale, zero = d["quantization"]
                    feeds[j] = (feeds[j].astype(np.float32) - zero) * scale
        return feeds[0].reshape(-1, 85)
