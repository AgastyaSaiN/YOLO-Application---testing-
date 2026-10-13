"""Loading and running the models: ONNX Runtime, OpenVINO and TFLite, plus the shared pre- and post-processing.

    model = load_model(entry, base_dir)          # entry = one item of models/catalog.json
    dets, ms = model.detect(frame_bgr, conf=0.25, iou=0.45)
    dets: array of [x1, y1, x2, y2, confidence, class_id] in the frame's pixel coordinates; ms = {"pre", "infer", "post"} in milliseconds.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THREADS = 4
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra "
        "giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup "
        "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
        "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").replace("_", " ").split()


def letterbox(img, size):
    """Resize keeping the aspect ratio and pad to size x size (grey), as the benchmark did. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def decode(pred, box_scale, conf, iou, r, pad, shape):
    """pred (N, 85) = x, y, w, h, objectness, 80 class scores -> detections in the original frame (one class per box, then NMS)."""
    pred = pred.reshape(-1, 85).astype(np.float32)
    cls_scores = pred[:, 5:]
    cls = cls_scores.argmax(1)
    score = pred[:, 4] * cls_scores[np.arange(len(pred)), cls]
    keep = score > conf
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    p, score, cls = pred[keep], score[keep], cls[keep]
    xywh = p[:, :4] * box_scale                                     # box_scale: 1 for pixel boxes, input size for 0-1 boxes
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1)    # x, y, w, h
    idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), score.tolist(), cls.tolist(), conf, iou)
    idx = np.array(idx, dtype=int).reshape(-1)
    if not idx.size:
        return np.zeros((0, 6), np.float32)
    b = boxes[idx]
    out = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3], score[idx], cls[idx]], 1).astype(np.float32)
    out[:, [0, 2]] = ((out[:, [0, 2]] - pad[0]) / r).clip(0, shape[1])
    out[:, [1, 3]] = ((out[:, [1, 3]] - pad[1]) / r).clip(0, shape[0])
    return out


class Model:
    """Base class: subclasses implement _load() and _run(x) where x is a float32 NCHW image batch (1, 3, S, S) scaled 0-1."""

    def __init__(self, entry, base_dir):
        self.entry, self.size, self.box_scale = entry, int(entry["input_size"]), float(entry["box_scale"])
        self.path = Path(base_dir) / "models" / entry["path"]
        self._load()

    def detect(self, frame_bgr, conf=0.25, iou=0.45):
        t0 = time.perf_counter()
        img, r, pad = letterbox(frame_bgr, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        t1 = time.perf_counter()
        pred = self._run(x)
        t2 = time.perf_counter()
        dets = decode(pred, self.box_scale, conf, iou, r, pad, frame_bgr.shape[:2])
        t3 = time.perf_counter()
        return dets, {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}

    def close(self):
        pass


class OnnxModel(Model):
    def _load(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def _run(self, x):
        return self.sess.run(None, {self.input: x})[0]


class OpenVinoModel(Model):
    def _load(self):
        import openvino as ov
        self.comp = ov.Core().compile_model(str(self.path), "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": THREADS})

    def _run(self, x):
        return next(iter(self.comp(x).values()))


class TFLiteModel(Model):
    """A TFLite model is a folder of stage files. TensorFlow is only installed in the 'Day 2' environment, so if this Python has no
    TensorFlow, the stages are run by a small helper process (tflite_worker.py) started with that environment's Python."""

    def _load(self):
        self.proc = None
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow  # noqa: F401
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from tflite_worker import Pipeline
            self.pipe = Pipeline(self.path, THREADS)
        except ImportError:
            py = os.environ.get("TFLITE_PYTHON") or str(Path(__file__).resolve().parent.parent / "Day 2" / ".venv" / "Scripts" / "python.exe")
            if not Path(py).exists():
                raise RuntimeError("TFLite models need TensorFlow. Install it (pip install tensorflow==2.15.1) or set TFLITE_PYTHON to a Python that has it (see README).")
            self.pipe = None
            env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
            self.proc = subprocess.Popen([py, str(Path(__file__).resolve().parent / "tflite_worker.py"), str(self.path), str(self.size), str(THREADS)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
            if self.proc.stdout.read(5) != b"READY":
                raise RuntimeError("The TFLite helper process failed to start.")

    def _run(self, x):
        if self.pipe is not None:
            return self.pipe(x)
        self.proc.stdin.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        self.proc.stdin.flush()
        n = int.from_bytes(self.proc.stdout.read(4), "little")
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("The TFLite helper process stopped.")
            buf += chunk
        return np.frombuffer(buf, np.float32).reshape(-1, 85)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            self.proc = None


def load_model(entry, base_dir):
    cls = {"onnxruntime": OnnxModel, "openvino": OpenVinoModel, "tflite": TFLiteModel}[entry["runtime"]]
    return cls(entry, base_dir)


def draw(frame, dets, thickness=2):
    """Draw boxes and labels on a copy of the frame."""
    out = frame.copy()
    for x1, y1, x2, y2, c, k in dets:
        k = int(k)
        col = tuple(int(v) for v in np.random.RandomState(k * 7 + 3).randint(60, 255, 3))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, thickness)
        label = f"{COCO[k]} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(int(y1) - th - 6, 0)
        cv2.rectangle(out, (int(x1), y0), (int(x1) + tw + 4, y0 + th + 6), col, -1)
        cv2.putText(out, label, (int(x1) + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
