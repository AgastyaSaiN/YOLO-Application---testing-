"""Copy the latest models from the benchmark folder (../Day 2) into ./models and write models/catalog.json and MODELS.md.

Run once (needs the "Day 2" folder next to this one):   python tools/make_catalog.py
The app only reads models/catalog.json, so afterwards it does not need the Day 2 folder.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DAY2 = HERE.parent / "Day 2"
MODELS = HERE / "models"
R = DAY2 / "results"
load = lambda n: json.loads((R / n).read_text())
acc = {**load("accuracy.json"), **load("accuracy_tflite.json"), **load("accuracy_ov.json")}
lat, mem = load("latency.json"), load("memory.json")
BASE = acc["m0_fp32"]["mAP50-95"]
DESIGN = {"m0": "FP32 (no quantization)", "m1": "M1: all INT8", "m2": "M2: backbone INT8, neck + head FP32", "m3": "M3: backbone + neck INT8, head FP32"}
entries, jobs = [], []          # jobs: (source, destination) copies


def add(group, runtime, rel_path, label, size, box_scale, note, acc_key=None, lat_key=None):
    """acc_key / lat_key point into the benchmark result files (None = not measured)."""
    a = acc.get(acc_key) if acc_key else None
    l = lat.get(lat_key) if lat_key else None
    m = mem.get(lat_key) if lat_key else None
    entries.append({
        "id": rel_path.replace("/", "_").replace(".onnx", "").replace(".xml", ""), "group": group, "runtime": runtime, "path": rel_path, "label": label,
        "input_size": size, "box_scale": box_scale, "note": note,
        "mAP50_95": round(a["mAP50-95"], 4) if a else None, "retained_pct": round(a["mAP50-95"] / BASE * 100, 1) if a else None,
        "mAP50": round(a["mAP50"], 4) if a else None, "file_mb": round(a["size_mb"], 2) if a else None,
        "bench_ms": round(l["latency_ms_median"], 1) if l else None, "bench_mem_mb": round(m["peak_mb"]) if m else None,
    })


# ---------------------------------------------------------------- ONNX Runtime
jobs.append((DAY2 / "models/fp32/yolov5n.onnx", MODELS / "onnxruntime/fp32_640.onnx"))
add("FP32 baseline", "onnxruntime", "onnxruntime/fp32_640.onnx", "FP32 baseline | 640 px", 640, 1.0, "Original YOLOv5n, no quantization. Boxes in pixels.", "m0_fp32", "m0_fp32")
jobs.append((DAY2 / "models/fp32_512/f512_norm.onnx", MODELS / "onnxruntime/fp32_512.onnx"))
add("FP32 baseline", "onnxruntime", "onnxruntime/fp32_512.onnx", "FP32 | 512 px input", 512, 512.0, "Original network exported at a smaller 512 px input, no quantization.", "f512_fp32", None)
for stem in ("m1_all_int8_pc", "m1_all_int8_pt", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"):
    d, gran = stem[:2], ("per-channel" if stem.endswith("_pc") else "per-tensor")
    name = f"int8_{stem}_512.onnx"
    jobs.append((DAY2 / "models/onnx_int8_512" / f"{stem}.onnx", MODELS / "onnxruntime" / name))
    add("INT8, 512 px input", "onnxruntime", f"onnxruntime/{name}", f"{DESIGN[d].split(':')[0]} {gran} | 512 px", 512, 512.0, DESIGN[d] + f", {gran} quantization.", f"s512_{stem}", f"s512_{stem}")

# ---------------------------------------------------------------- OpenVINO
add("FP32 baseline", "openvino", "onnxruntime/fp32_640.onnx", "FP32 baseline | 640 px (run in OpenVINO)", 640, 1.0, "The FP32 ONNX model run by OpenVINO.", "m0_fp32", "ov_m0_fp32")
for i in (1, 2, 3):
    stem = f"f512_m{i}_mixed"
    for ext in (".xml", ".bin"):
        jobs.append((DAY2 / "models/ov_int8" / (stem + ext), MODELS / "openvino" / (stem + ext)))
    add("INT8, 512 px input", "openvino", f"openvino/{stem}.xml", f"M{i} | OpenVINO INT8 | 512 px", 512, 512.0, DESIGN[f"m{i}"] + ". INT8 with OpenVINO NNCF.", "ovq_" + stem, "ovq_" + stem)

# ---------------------------------------------------------------- TFLite (one .tflite file per folder)
for d in sorted((DAY2 / "models/tflite").glob("*")):
    if d.name.startswith("_") or not (d / "manifest.json").exists():
        continue
    key = d.name
    if not (key.startswith(("s512_", "t512_")) or key == "m0_fp32"):
        continue
    size = 640 if key == "m0_fp32" else 512
    base = key.replace("s512_", "").replace("t512_", "")
    io8 = base.endswith("_io8")
    base = base.removesuffix("_io8")
    k = base[:2]
    gran = "" if k == "m0" else (", per-channel" if base.endswith("_pc") else ", per-tensor")
    extra = ", integer input/output" if io8 else ""
    jobs.append((d, MODELS / "tflite" / d.name))
    add("TFLite" if size == 512 else "FP32 baseline", "tflite", f"tflite/{d.name}", f"TFLite {DESIGN[k].split(':')[0]}{gran}{extra} | {size} px", size, float(size),
        DESIGN[k] + gran + extra + ". A single .tflite file.", f"tflite_{key}", f"tflite_{key}")

for src, dst in jobs:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy2(src, dst)
(MODELS / "catalog.json").write_text(json.dumps(entries, indent=1))

# ---------------------------------------------------------------- MODELS.md
lines = ["# The models in this folder", "",
         "These are the latest models: the FP32 reference and everything at the current 512 px input size, in three runtimes. Every model is YOLOv5n (nano) trained on COCO (80 classes). "
         "**The Detect head (the last layer) is unchanged in every model**; the models differ in how much of the rest of the network is INT8, which runtime runs them and their input size.", "",
         "How to read the names:", "- **M1** all INT8 | **M2** backbone INT8, neck and head stay FP32 | **M3** backbone and neck INT8, head stays FP32.",
         "- **per-channel / per-tensor**: how finely the INT8 scales are chosen (per-channel is more accurate).",
         "- **640 px / 512 px**: the picture size the model looks at. The FP32 reference uses the full 640 px; all INT8 models here use the smaller 512 px, which is faster and a little less accurate.",
         "- **integer input/output**: the model takes and gives 8-bit integers, as integer-only accelerators usually need.",
         "- **Accuracy** is mAP@0.5:0.95 on all 5000 COCO val2017 images; **retained** is the share of the FP32 model's accuracy.",
         "- **Benchmark ms** is the model time per image on the benchmark laptop (4 pinned cores, batch 1). It will differ on your machine.", ""]
groups = []
for e in entries:
    if e["group"] not in groups:
        groups.append(e["group"])
for g in groups:
    lines += [f"## {g}", "", "| Model | Runtime | File | Input | mAP50-95 | Retained | Size (MB) | Benchmark ms |", "|---|---|---|---|---|---|---|---|"]
    for e in [x for x in entries if x["group"] == g]:
        f = lambda v, s="{}": "-" if v is None else s.format(v)
        lines.append(f"| {e['label'].replace('|', '-')} | {e['runtime']} | `{e['path']}` | {e['input_size']} | {f(e['mAP50_95'])} | {f(e['retained_pct'], '{}%')} | {f(e['file_mb'])} | {f(e['bench_ms'])} |")
    lines.append("")
(HERE / "MODELS.md").write_text("\n".join(lines), encoding="utf-8")
total = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file()) / 1e6
print(f"{len(entries)} models listed, {total:.0f} MB in {MODELS}")
