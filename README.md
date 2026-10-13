# Yolo thing

Work is organised **one folder per day**. Each day is self-contained: its own scripts, models, results and docs, with its own README. A finished day is treated as frozen, and new work goes in the next day's folder.

| Folder | Status | What it holds |
|---|---|---|
| [Day 1](Day%201/README.md) | Done | YOLOv5n INT8 quantization with the Detect head kept in FP32, five variants, COCO benchmark, report. Result: about half the size, 19% faster, -0.9% accuracy. |
| [Day 2](Day%202/README.md) | Done | Original YOLOv5n, three INT8 designs x per-channel/per-tensor x ONNX/TFLite, full COCO benchmark with confidence intervals, layer sensitivity, calibration study. [Report](Day%202/results/report.html). |

## Conventions

- **Self-contained days.** Scripts find their files relative to their own folder, so a day can be moved or copied as a unit.
- **Reuse, don't copy.** If Day 2 needs a Day 1 asset (the original model, the COCO data), refer to it by path, for example `../Day 1/models/original/yolov5nu.onnx`. This avoids duplicating the 1.1 GB dataset.
- **Don't edit a finished day.** If a Day 1 result needs fixing, note it in Day 2 rather than quietly changing the old numbers.
- **Each day's README** states the goal, the result, and how to re-run it.
