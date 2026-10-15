"""Loading and running the models: ONNX Runtime, OpenVINO and TFLite, plus the shared pre- and post-processing.

    model = load_model(entry, base_dir)          # entry = one item of models/catalog.json
    dets, ms = model.detect(frame_bgr, conf=0.25, iou=0.45)
    dets: array of [x1, y1, x2, y2, confidence, class_id] in the frame's pixel coordinates; ms = {"pre", "infer", "post"} in milliseconds.
"""
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

    def _prepare(self, img_bgr):
        """The letterboxed 8-bit picture as the network input: float32, RGB, 0-1, NCHW."""
        return np.ascontiguousarray(img_bgr[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def detect(self, frame_bgr, conf=0.25, iou=0.45):
        t0 = time.perf_counter()
        img, r, pad = letterbox(frame_bgr, self.size)
        x = self._prepare(img)
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
    """A TFLite model (a .tflite file, or a folder holding one), run with LiteRT. Integer-input models get the picture as 8-bit integers directly."""

    def _load(self):
        from tflite_runner import Pipeline
        self.pipe = Pipeline(self.path, THREADS)
        d = self.pipe.stages[0][1][0]
        self.lut = None
        if d["dtype"] != np.float32:                  # integer-input model: a 256-entry table turns the 8-bit picture straight into the model's integer input
            scale, zero = d["quantization"]
            info = np.iinfo(d["dtype"])
            self.lut = np.clip(np.round(np.arange(256) / 255.0 / scale) + zero, info.min, info.max).astype(d["dtype"])

    def _prepare(self, img_bgr):
        if self.lut is not None:
            return np.ascontiguousarray(self.lut[img_bgr[:, :, ::-1]][None])          # 1 x H x W x 3, integer
        return super()._prepare(img_bgr)

    def _run(self, x):
        return self.pipe(x)


def load_model(entry, base_dir):
    cls = {"onnxruntime": OnnxModel, "openvino": OpenVinoModel, "tflite": TFLiteModel}[entry["runtime"]]
    return cls(entry, base_dir)


def entry_from_file(path):
    """Describe a model file the user picked (.tflite, .onnx or .xml) so it can be run like the listed models: runtime from the extension,
    input size from the model's input shape, and whether its boxes are in pixels or 0-1 by looking at what it outputs for a plain grey picture."""
    path = Path(path).resolve()
    ext = path.suffix.lower()
    if ext == ".tflite":
        runtime = "tflite"
    elif ext == ".onnx":
        runtime = "onnxruntime"
    elif ext == ".xml":
        runtime = "openvino"
    else:
        raise ValueError("Choose a .tflite, .onnx or .xml file.")
    name = path.parent.name if path.stem == "stage1" else path.stem            # a .tflite file named stage1 belongs to the model folder it is in
    entry = {"id": name, "group": "Added by you", "runtime": runtime, "path": str(path), "label": name, "input_size": 640, "box_scale": 1.0,
             "note": f"Model file you added: {path.name}", "mAP50_95": None, "retained_pct": None, "mAP50": None,
             "file_mb": round(path.stat().st_size / 1e6, 2), "bench_ms": None, "bench_mem_mb": None}
    if runtime == "onnxruntime":
        import onnxruntime as ort
        shape = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape          # [1, 3, H, W]
    elif runtime == "openvino":
        import openvino as ov
        shape = list(ov.Core().read_model(str(path)).input(0).shape)
    else:
        from tflite_runner import Pipeline
        h, w = Pipeline(path, 1).stages[0][1][0]["shape"][1:3]                                                       # NHWC
        shape = [1, 3, int(h), int(w)]
    if not all(isinstance(v, int) for v in shape[2:]) or shape[2] != shape[3]:
        raise ValueError(f"The model needs a fixed square input; this one has input shape {shape}.")
    entry["input_size"] = int(shape[2])
    m = load_model(entry, ".")
    try:
        grey = np.full((entry["input_size"], entry["input_size"], 3), 114, np.uint8)
        pred = np.asarray(m._run(m._prepare(grey))).reshape(-1, 85)
        if pred.shape[1] != 85:
            raise ValueError("This does not look like a YOLOv5 detector (it should output rows of 85 values).")
        entry["box_scale"] = 1.0 if float(pred[:, 0].max()) > 2.0 else float(entry["input_size"])        # x positions up to the picture size mean pixels, up to 1 mean fractions
    finally:
        m.close()
    return entry


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
