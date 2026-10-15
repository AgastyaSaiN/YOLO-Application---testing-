# Real-time model tester

A small desktop program for trying the YOLOv5n models on a **webcam** or on a **video file**. It draws the detections on the picture and shows a short analysis (speed, number of objects, confidence, accuracy of the chosen model) next to it. With a video file it runs **several models one after the other** on the same footage and puts them side by side in a table.

It is meant for looking at the models we built (FP32, INT8 in three designs, different input sizes, three runtimes) and getting a feel for how they behave, not for benchmarking. For the careful measurements see the benchmark reports.

---

## 1. Quick start

You need Python 3.10 or newer on Windows (the window uses `tkinter`, which comes with Python).

```
cd "Real time testing"
pip install -r requirements.txt
python app.py
```

You can also double-click `run_app.bat`.

1. In the model list on the left, click a model (Ctrl-click or Shift-click to pick several).
2. Choose **Webcam** or **Video file** at the top (for a video press **Browse...**; `sample_video/demo.mp4` is included).
3. Press **Start**.

To check that everything works without opening a window:

```
python app.py --selftest
```

This runs four models on `sample_video/demo.mp4` and prints a small table of speed and detections.

---

## 2. The window

```
+--------------------------------------------------------------------------------------+
| ( ) Webcam [0]  ( ) Video file [Browse..]  Confidence  Overlap  Every Nth  [ ] Save  |
|                                                         [Start] [Stop]               |
+------------------+------------------------------------------+------------------------+
| Models           |                                          | Analysis               |
| [filter v]       |                                          |  model description     |
|  [ORT] FP32 ...  |        video with boxes                  |  live speed            |
|  [ORT] M1 ...    |                                          |  model-time graph      |
|  [OV]  M3 ...    |                                          |  objects in this frame |
|  [TFL] ...       |   status line                            |  video comparison table|
+------------------+------------------------------------------+------------------------+
```

### Top bar

| Control | What it does |
|---|---|
| **Webcam** + number | Uses the camera with that index. 0 is normally the built-in camera; try 1 or 2 for an external one. |
| **Video file** + **Browse...** | Uses a video file (mp4, avi, mov, mkv, wmv). |
| **Confidence** | Detections whose confidence is below this are hidden. Default 0.25. Higher shows fewer, surer detections; lower shows more, including doubtful ones. |
| **Overlap (IoU)** | When two boxes of the same class overlap more than this, only the better one is kept. Default 0.45. |
| **Every Nth frame** | Video only. 1 processes every frame; 5 processes every fifth, which finishes long videos faster. |
| **Save annotated videos** | Video only. Writes one video with the boxes drawn per model into the `outputs/` folder, named after the model. |
| **Start / Stop** | Starts or stops the run. Confidence and Overlap can be changed while it runs. |

### Model list (left)

- Every model is listed with its short runtime tag: **[ORT]** ONNX Runtime, **[OV]** OpenVINO, **[TFL]** TFLite.
- **mAP** is the accuracy measured on all 5000 COCO val2017 images (mAP@0.5:0.95; higher is better; the original FP32 model scores 0.281). **ms** is the time the model needed for one image on the benchmark laptop; it will be different on your computer. A dash means that figure was not measured for that model.
- The drop-down above the list shows only one runtime, or only the INT8 models.
- Clicking a model also shows its description in the Analysis panel.

### Webcam mode

Runs the **first selected model** live on the camera. While it runs you can click another model in the list; the program loads it and switches without stopping the camera. The camera is asked for a 640 x 480 picture to keep things fast.

### Video mode

Runs **every selected model, one after the other, over the whole video** (from the top of the list to the bottom of the selection). After each model finishes, a row is added to the **Video comparison** table. When all are done the status line says so. **Save comparison as CSV** writes the table to a file you choose.

### Analysis panel (right)

| Part | Meaning |
|---|---|
| Model description | Name, what it is, input size, file size, accuracy (mAP@0.5:0.95 and the share of the FP32 model's accuracy it keeps) and its benchmark time. |
| Live line | **frames/s** is 1000 divided by the total time per frame (resize + model + box decoding; drawing and display are not counted). **Model** is the time the network took. **resize** is the time to prepare the picture. **boxes** is the time to turn the raw output into boxes. In video mode it also shows the frame number. |
| Detections / mean confidence | Number of objects found in the current frame and their average confidence. |
| Graph | The model time of the last 100 frames. A flat line means steady speed; spikes usually mean something else was using the computer. |
| Objects in this frame | Count per class, for example `person: 3, chair: 2`. |
| Video comparison | One row per model for the whole video: **Frames** processed, **ms/img** average total time per frame, **FPS** frames per second, **Dets** average detections per frame, **mAP** the model's benchmark accuracy. |

Reading the comparison: a faster model with fewer **Dets** than the others may be missing objects; a model with a clearly higher **Dets** may be adding false ones. The mAP column tells you which one to trust. All models see the same footage and the same confidence setting, so the numbers are comparable with each other.

---

## 3. The models

The `models/` folder contains the latest models, 22 in total, and `models/catalog.json` is the list the program reads. **`MODELS.md` describes each model** (accuracy, size, benchmark speed) and explains the naming. In short:

| Part of the name | Meaning |
|---|---|
| **FP32** | The original network without quantization (full-precision numbers). |
| **M1** / **M2** / **M3** | INT8 (8-bit) designs. M1: everything INT8. M2: backbone INT8, neck and head kept FP32. M3: backbone and neck INT8, head kept FP32. |
| **per-channel** / **per-tensor** | How finely the INT8 scales are chosen. Per-channel is more accurate. |
| **640 / 512 px** | The picture size the network looks at. The FP32 reference uses the full 640 px; all INT8 models here use the smaller 512 px, which is faster and a little less accurate. |

Every model is YOLOv5n (the nano size) trained on COCO and detects the same 80 classes. **The final detection layer (Detect head) is unchanged in every model.**

Only the latest models are kept here: the FP32 reference and everything at the current 512 px input size, in the three runtimes. The earlier 640 px INT8 models, the 448 and 416 px sizes and the experiments (ReLU-activation models, fine-tuned variants) are in the benchmark folder (`..\Day 2\models`) and are described in its reports.

```
models/
├── catalog.json        the list the program reads (name, runtime, file, input size, accuracy, speed)
├── onnxruntime/        8 .onnx files (one file per model)
├── openvino/           3 models, each a .xml (graph) + .bin (weights) pair
└── tflite/             10 folders, each holding one .tflite file + manifest.json
```

### Runtimes

| Runtime | Files | Needs |
|---|---|---|
| ONNX Runtime | one `.onnx` file | `onnxruntime` |
| OpenVINO | `.xml` + `.bin` (keep them together) | `openvino` |
| TFLite | a folder with one `.tflite` file | `ai-edge-litert` (see below) |

All run on the CPU with 4 threads. The OpenVINO entry "FP32 baseline - 640 px (run in OpenVINO)" is the same ONNX file as the ONNX Runtime baseline, run by OpenVINO instead.

### TFLite models

Each TFLite model is one `.tflite` file in its own folder (with a small `manifest.json`). They run with **LiteRT** (`pip install ai-edge-litert`, listed in `requirements.txt`), Google's current TFLite runtime, which runs INT8 models noticeably faster than the interpreter inside TensorFlow 2.15 (the same files run about 1.4 to 1.7 times faster; see the benchmark report). TensorFlow itself is not needed.

The two models marked **integer input/output** take and give 8-bit integers, as integer-only accelerators (NPUs) usually need. The program turns the picture into 8-bit integers directly with a lookup table, so they run at about the same speed as the others.

---

## 4. How a frame is processed

1. **Resize.** The frame is shrunk to fit a square of the model's input size and padded with grey, keeping the aspect ratio (this is what the benchmark did).
2. **Model.** The picture goes through the network. The output is a long list of candidate boxes, each with a box, an "is there an object" score and 80 class scores.
3. **Boxes.** A candidate's confidence is object score x best class score. Candidates below the confidence setting are dropped; the remaining ones go through non-maximum suppression (overlapping boxes of the same class are merged). The boxes are mapped back to the original frame's size.
4. **Draw.** Boxes and labels are drawn; the numbers on the right are computed from the same result.

Some models give box positions in pixels and some as fractions of the picture (the `box_scale` field in `catalog.json` takes care of this). The code is in `engines.py` (about 170 lines) and `app.py` (the window).

---

## 5. Using your own video, or adding a model

- **Your own video:** press **Browse...** and choose it. Any file OpenCV can read works.
- **Another model of the same kind, quickest way:** press **Add model file...** under the model list and choose a `.tflite`, `.onnx` or `.xml` file (for TFLite the `.tflite` file itself). The program reads the input size from the model and works out whether its boxes are in pixels or fractions of the picture by looking at what it outputs for a grey picture; the model is added to the list for this session and selected. It must be a YOLOv5-style detector with a fixed square input and rows of 85 values. Models added this way show no accuracy or benchmark figures.
- **To keep a model in the list permanently**, copy its file(s) into `models/` and add one entry to `models/catalog.json`:

  ```json
  {"id": "my_model", "group": "Mine", "runtime": "onnxruntime", "path": "onnxruntime/my_model.onnx",
   "label": "My model", "input_size": 640, "box_scale": 640.0, "note": "What it is.",
   "mAP50_95": null, "retained_pct": null, "mAP50": null, "file_mb": null, "bench_ms": null, "bench_mem_mb": null}
  ```

  `runtime` is `onnxruntime`, `openvino` (path to the `.xml`) or `tflite` (path to the folder). The model must take a 1 x 3 x S x S float32 picture scaled 0-1 (RGB) and output the standard YOLOv5 table of N x 85 values. Use `box_scale` 1.0 if the boxes are in pixels and S if they are fractions of the picture.
- `tools/make_catalog.py` is the script that copied the models from the benchmark folder and wrote `catalog.json` and `MODELS.md`. It is not needed to run the program.

---

## 6. If something goes wrong

| Problem | What to try |
|---|---|
| "Could not open the webcam" | Change the number next to Webcam (0, 1, 2), close other programs that use the camera (video calls, camera apps), check the Windows camera privacy setting for desktop apps. |
| Webcam shows the picture but nothing is detected | Lower **Confidence**; make sure the scene has COCO objects (people, cups, chairs, laptops...); try the FP32 or an M2 model; the accuracy column shows which are the most accurate. |
| A TFLite model is skipped | LiteRT is not installed in this Python: `pip install ai-edge-litert`. |
| An OpenVINO model fails to load | The `.xml` and `.bin` files must be in the same folder with the same name. |
| It is slow | Choose a 512 px model (instead of the 640 px FP32 one) or an OpenVINO model; raise **Every Nth frame** for videos; close other heavy programs. Speed depends on your CPU. |
| `No module named ...` | `pip install -r requirements.txt`. The program needs `numpy` below version 2 together with the OpenCV and OpenVINO versions in the file; if you see NumPy errors, `pip install "numpy<2"`. |
| The window is blurry or too big | Windows display scaling; the program asks for sharp rendering, but you can resize or maximise the window. |

---

## 7. Limits to keep in mind

- The speeds shown include everything the program does around the model on **your** computer, and the benchmark times in the list were measured on a different laptop with pinned cores, so they will not match.
- These are small models. Expect misses on small, far away or unusual objects, and some wrong labels, especially with the 512 px models.
- Only the CPU is used.
- The program uses the camera and files you choose; nothing is sent anywhere.
