# The models in this folder

These are the latest models: the FP32 reference and everything at the current 512 px input size, in three runtimes. Every model is YOLOv5n (nano) trained on COCO (80 classes). **The Detect head (the last layer) is unchanged in every model**; the models differ in how much of the rest of the network is INT8, which runtime runs them and their input size.

How to read the names:
- **M1** all INT8 | **M2** backbone INT8, neck and head stay FP32 | **M3** backbone and neck INT8, head stays FP32.
- **per-channel / per-tensor**: how finely the INT8 scales are chosen (per-channel is more accurate).
- **640 px / 512 px**: the picture size the model looks at. The FP32 reference uses the full 640 px; all INT8 models here use the smaller 512 px, which is faster and a little less accurate.
- **integer input/output**: the model takes and gives 8-bit integers, as integer-only accelerators usually need.
- **Accuracy** is mAP@0.5:0.95 on all 5000 COCO val2017 images; **retained** is the share of the FP32 model's accuracy.
- **Benchmark ms** is the model time per image on the benchmark laptop (4 pinned cores, batch 1). It will differ on your machine.

## FP32 baseline

| Model | Runtime | File | Input | mAP50-95 | Retained | Size (MB) | Benchmark ms |
|---|---|---|---|---|---|---|---|
| FP32 baseline - 640 px | onnxruntime | `onnxruntime/fp32_640.onnx` | 640 | 0.2811 | 100.0% | 7.92 | 27.9 |
| FP32 - 512 px input | onnxruntime | `onnxruntime/fp32_512.onnx` | 512 | 0.2726 | 97.0% | 7.78 | - |
| FP32 baseline - 640 px (run in OpenVINO) | openvino | `onnxruntime/fp32_640.onnx` | 640 | 0.2811 | 100.0% | 7.92 | 23.4 |
| TFLite FP32 (no quantization) - 640 px | tflite | `tflite/m0_fp32` | 640 | 0.2811 | 100.0% | 7.93 | 27.6 |

## INT8, 512 px input

| Model | Runtime | File | Input | mAP50-95 | Retained | Size (MB) | Benchmark ms |
|---|---|---|---|---|---|---|---|
| M1 per-channel - 512 px | onnxruntime | `onnxruntime/int8_m1_all_int8_pc_512.onnx` | 512 | 0.2522 | 89.7% | 2.28 | 17.3 |
| M1 per-tensor - 512 px | onnxruntime | `onnxruntime/int8_m1_all_int8_pt_512.onnx` | 512 | 0.2426 | 86.3% | 2.21 | 17.0 |
| M2 per-channel - 512 px | onnxruntime | `onnxruntime/int8_m2_backbone_int8_pc_512.onnx` | 512 | 0.2694 | 95.8% | 4.8 | 14.4 |
| M2 per-tensor - 512 px | onnxruntime | `onnxruntime/int8_m2_backbone_int8_pt_512.onnx` | 512 | 0.2616 | 93.1% | 4.76 | 14.4 |
| M3 per-channel - 512 px | onnxruntime | `onnxruntime/int8_m3_backbone_neck_int8_pc_512.onnx` | 512 | 0.2668 | 94.9% | 2.78 | 16.2 |
| M3 per-tensor - 512 px | onnxruntime | `onnxruntime/int8_m3_backbone_neck_int8_pt_512.onnx` | 512 | 0.2562 | 91.1% | 2.71 | 16.4 |
| M1 - OpenVINO INT8 - 512 px | openvino | `openvino/f512_m1_mixed.xml` | 512 | 0.2669 | 95.0% | 3.13 | 10.3 |
| M2 - OpenVINO INT8 - 512 px | openvino | `openvino/f512_m2_mixed.xml` | 512 | 0.2713 | 96.5% | 5.39 | 12.1 |
| M3 - OpenVINO INT8 - 512 px | openvino | `openvino/f512_m3_mixed.xml` | 512 | 0.2692 | 95.7% | 3.46 | 10.6 |

## TFLite

| Model | Runtime | File | Input | mAP50-95 | Retained | Size (MB) | Benchmark ms |
|---|---|---|---|---|---|---|---|
| TFLite FP32 (no quantization) - 512 px | tflite | `tflite/s512_m0_fp32` | 512 | 0.2726 | 97.0% | 7.79 | 18.7 |
| TFLite M1, per-channel - 512 px | tflite | `tflite/t512_m1_all_int8_pc` | 512 | 0.2469 | 87.8% | 2.15 | 10.7 |
| TFLite M1, per-channel, integer input/output - 512 px | tflite | `tflite/t512_m1_all_int8_pc_io8` | 512 | 0.2469 | 87.8% | 2.15 | 17.3 |
| TFLite M1, per-tensor - 512 px | tflite | `tflite/t512_m1_all_int8_pt` | 512 | 0.2385 | 84.8% | 2.02 | 10.3 |
| TFLite M1, per-tensor, integer input/output - 512 px | tflite | `tflite/t512_m1_all_int8_pt_io8` | 512 | 0.2385 | 84.8% | 2.02 | 17.3 |
| TFLite M2, per-channel - 512 px | tflite | `tflite/t512_m2_backbone_int8_pc` | 512 | 0.2677 | 95.2% | 4.73 | 13.8 |
| TFLite M2, per-tensor - 512 px | tflite | `tflite/t512_m2_backbone_int8_pt` | 512 | 0.261 | 92.8% | 4.67 | 13.8 |
| TFLite M3, per-channel - 512 px | tflite | `tflite/t512_m3_backbone_neck_int8_pc` | 512 | 0.2643 | 94.0% | 2.67 | 11.1 |
| TFLite M3, per-tensor - 512 px | tflite | `tflite/t512_m3_backbone_neck_int8_pt` | 512 | 0.2545 | 90.5% | 2.56 | 11.2 |
