"""Runs a TFLite pipeline (a folder with manifest.json and stage1.tflite, stage2.tflite, ...).

Used in two ways:
  * imported:  from tflite_worker import Pipeline          (when TensorFlow is installed in the same Python as the app)
  * as a helper process started by engines.py with a Python that has TensorFlow:
        python tflite_worker.py <pipeline folder> <input size> <threads>
    It prints READY, then for every image sent on stdin (float32, 1 x 3 x size x size, 0-1) it writes back the raw detections
    (4-byte length, then float32 values).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


class Pipeline:
    """Stage k's outputs feed stage k+1's inputs, matched by tensor shape. INT8 stages take and give float tensors at their edges."""

    def __init__(self, folder, threads=4):
        self.folder = Path(folder)
        man = json.loads((self.folder / "manifest.json").read_text())
        self.stages = []
        for name in man["stages"]:
            it = tf.lite.Interpreter(model_path=str(self.folder / name), num_threads=threads)
            it.allocate_tensors()
            self.stages.append((it, it.get_input_details(), it.get_output_details()))
        # layout hints written when the models were built: some stage inputs must be transposed back (see the project's docs)
        self.input_perm = {(h["stage"], tuple(h["shape"])): tuple(h["perm"]) for h in man.get("input_perm", [])}

    def __call__(self, x_nchw):
        feeds = [np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1), dtype=np.float32)]        # NCHW -> NHWC
        for k, (it, ins, outs) in enumerate(self.stages, 1):
            for d in ins:
                v = next(f for f in feeds if tuple(f.shape) == tuple(d["shape"]))
                perm = self.input_perm.get((k, tuple(d["shape"])))
                if perm:
                    v = np.ascontiguousarray(v.transpose(perm))
                it.set_tensor(d["index"], v)
            it.invoke()
            feeds = [it.get_tensor(d["index"]) for d in outs]
        return feeds[0].reshape(-1, 85)


if __name__ == "__main__":
    folder, size, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pipe = Pipeline(folder, threads)
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    out.write(b"READY"); out.flush()
    need = 3 * size * size * 4
    while True:
        buf = inp.read(need)
        if len(buf) < need:
            break
        y = np.ascontiguousarray(pipe(np.frombuffer(buf, np.float32).reshape(1, 3, size, size)), dtype=np.float32).tobytes()
        out.write(len(y).to_bytes(4, "little") + y); out.flush()
