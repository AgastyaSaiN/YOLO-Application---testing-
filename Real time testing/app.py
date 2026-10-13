"""YOLOv5n model tester: run any of the models on a webcam or a video file and see the detections and a little analysis.

    python app.py                      open the window
    python app.py --selftest           no window: run a few models on sample_video/demo.mp4 and print a table

Left: pick one or more models (the list comes from models/catalog.json).   Centre: the video with detections.   Right: analysis.
  * Webcam: runs the first selected model live. Click another model in the list to switch while it is running.
  * Video file: runs EVERY selected model one after the other on the same video and fills the comparison table on the right.
"""
import csv
import json
import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import engines

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "models" / "catalog.json").read_text())
SHORT = {"onnxruntime": "ORT", "openvino": "OV", "tflite": "TFL"}
for e in CATALOG:
    e["name"] = f"[{SHORT[e['runtime']]}] {e['label']}"


# ----------------------------------------------------------------------------------- the work (shared by the window and --selftest)
def run_video(model, path, conf, iou, step, on_frame, stop, writer_path=None):
    """Run one model over a whole video file. on_frame(annotated_bgr, stats) is called for every processed frame.
    Returns a summary dict (frames, ms per image, fps, average detections)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    writer, idx, n = None, -1, 0
    infer, total_ms, ndets = [], [], []
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        dets, t = model.detect(frame, conf, iou)
        out = engines.draw(frame, dets)
        n += 1
        infer.append(t["infer"]); total_ms.append(sum(t.values())); ndets.append(len(dets))
        on_frame(out, make_stats(model, dets, t, idx, total))
        if writer_path:
            if writer is None:
                writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src / step, (out.shape[1], out.shape[0]))
            writer.write(out)
    cap.release()
    if writer:
        writer.release()
    if not n:
        return None
    e = model.entry
    return {"model": e["name"], "frames": n, "infer_ms": float(np.mean(infer)), "total_ms": float(np.mean(total_ms)), "fps": 1000 / float(np.mean(total_ms)),
            "avg_dets": float(np.mean(ndets)), "mAP": e["mAP50_95"], "retained": e["retained_pct"], "file_mb": e["file_mb"], "bench_ms": e["bench_ms"]}


def make_stats(model, dets, t, idx=0, total=0):
    counts = Counter(engines.COCO[int(k)] for k in dets[:, 5])
    return {"model": model.entry["name"], "infer": t["infer"], "pre": t["pre"], "post": t["post"], "total": sum(t.values()), "ndet": len(dets),
            "avg_conf": float(dets[:, 4].mean()) if len(dets) else 0.0, "counts": counts, "frame": idx, "total_frames": total}


# ----------------------------------------------------------------------------------- the window
def main_window():
    try:                                          # sharp text on high-resolution screens (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("YOLOv5n model tester")
            self.geometry("1700x900")
            try:
                self.state("zoomed")              # start maximised on Windows
            except Exception:
                pass
            self.stop = threading.Event()
            self.worker = None
            self.lock = threading.Lock()
            self.latest = None                    # (annotated frame, stats) from the worker thread
            self.events = queue.Queue()           # messages from the worker thread to the window (Tk must only be touched by the main thread)
            self.rows, self.history, self.fps_ema = [], [], 0.0
            self.video_path = None
            self.wanted = None                    # webcam mode: entry the user clicked last
            self.status_text = "Pick a source and one or more models, then press Start."
            self.shown = list(CATALOG)
            self.cfg = {"conf": 0.25, "iou": 0.45, "step": 1, "save": False, "cam": 0}
            self.build()
            self.after(30, self.poll)
            self.protocol("WM_DELETE_WINDOW", self.quit_app)

        # ---------------- layout
        def build(self):
            top = ttk.Frame(self, padding=6)
            top.pack(fill="x")
            self.source = tk.StringVar(value="webcam")
            ttk.Radiobutton(top, text="Webcam", variable=self.source, value="webcam").pack(side="left")
            self.cam = tk.IntVar(value=0)
            ttk.Spinbox(top, from_=0, to=5, width=3, textvariable=self.cam).pack(side="left", padx=(2, 12))
            ttk.Radiobutton(top, text="Video file", variable=self.source, value="video").pack(side="left")
            ttk.Button(top, text="Browse...", command=self.browse).pack(side="left", padx=4)
            self.video_label = ttk.Label(top, text="(no file chosen)", width=28)
            self.video_label.pack(side="left")
            ttk.Label(top, text="  Confidence").pack(side="left")
            self.conf = tk.DoubleVar(value=0.25)
            ttk.Scale(top, from_=0.05, to=0.9, variable=self.conf, length=110).pack(side="left")
            self.conf_text = ttk.Label(top, text="0.25", width=4)
            self.conf_text.pack(side="left")
            ttk.Label(top, text=" Overlap (IoU)").pack(side="left")
            self.iou = tk.DoubleVar(value=0.45)
            ttk.Scale(top, from_=0.1, to=0.9, variable=self.iou, length=90).pack(side="left")
            ttk.Label(top, text="  Every Nth frame").pack(side="left")
            self.step = tk.IntVar(value=1)
            ttk.Spinbox(top, from_=1, to=30, width=3, textvariable=self.step).pack(side="left")
            self.save = tk.BooleanVar(value=False)
            ttk.Checkbutton(top, text="Save annotated videos", variable=self.save).pack(side="left", padx=10)
            self.start_btn = ttk.Button(top, text="Start", command=self.start)
            self.start_btn.pack(side="left", padx=4)
            ttk.Button(top, text="Stop", command=self.stop_run).pack(side="left")

            body = ttk.Frame(self)
            body.pack(fill="both", expand=True)
            # left: models
            left = ttk.Frame(body, padding=6)
            left.pack(side="left", fill="y")
            ttk.Label(left, text="Models (Ctrl / Shift to select several)").pack(anchor="w")
            self.filter = ttk.Combobox(left, state="readonly", width=34, values=["All models", "ONNX Runtime", "OpenVINO", "TFLite", "INT8 only"])
            self.filter.current(0)
            self.filter.pack(fill="x", pady=3)
            self.filter.bind("<<ComboboxSelected>>", lambda _e: self.fill_models())
            self.tree = ttk.Treeview(left, columns=("map", "ms"), selectmode="extended", height=28)
            self.tree.heading("#0", text="Model"); self.tree.heading("map", text="mAP"); self.tree.heading("ms", text="ms")
            self.tree.column("#0", width=330); self.tree.column("map", width=48, anchor="e"); self.tree.column("ms", width=40, anchor="e")
            sb = ttk.Scrollbar(left, command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="y"); sb.pack(side="left", fill="y")
            self.tree.bind("<<TreeviewSelect>>", self.on_select)
            self.fill_models()
            # centre: video
            mid = ttk.Frame(body, padding=6)
            mid.pack(side="left", fill="both", expand=True)
            self.frame_box = tk.Frame(mid, bg="black", width=640, height=480)       # fixed box: the picture is scaled to fit it
            self.frame_box.pack(fill="both", expand=True)
            self.frame_box.pack_propagate(False)
            self.video = tk.Label(self.frame_box, bg="black")
            self.video.place(relx=0.5, rely=0.5, anchor="center")
            self.status = ttk.Label(mid, text=self.status_text, wraplength=700)
            self.status.pack(fill="x", pady=4)
            # right: analysis
            right = ttk.Frame(body, padding=6, width=470)
            right.pack(side="left", fill="y")
            ttk.Label(right, text="Analysis", font=("Segoe UI", 12, "bold")).pack(anchor="w")
            self.info = tk.Text(right, width=58, height=8, wrap="word", bg="#f6f6f6", relief="flat")
            self.info.pack(fill="x", pady=3)
            self.live = tk.Text(right, width=58, height=8, wrap="word", bg="#f6f6f6", relief="flat")
            self.live.pack(fill="x", pady=3)
            ttk.Label(right, text="Model time per frame (ms, last 100)").pack(anchor="w")
            self.spark = tk.Canvas(right, width=440, height=60, bg="white", highlightthickness=1, highlightbackground="#ccc")
            self.spark.pack(fill="x", pady=3)
            ttk.Label(right, text="Objects in this frame").pack(anchor="w")
            self.objs = tk.Text(right, width=58, height=6, wrap="word", bg="#f6f6f6", relief="flat")
            self.objs.pack(fill="x", pady=3)
            ttk.Label(right, text="Video comparison (one row per model)").pack(anchor="w")
            cols = ("frames", "ms", "fps", "dets", "map")
            self.table = ttk.Treeview(right, columns=cols, height=7)
            for c, t, w in (("#0", "Model", 200), ("frames", "Frames", 52), ("ms", "ms/img", 52), ("fps", "FPS", 42), ("dets", "Dets", 40), ("map", "mAP", 48)):
                self.table.heading(c, text=t); self.table.column(c, width=w, anchor="w" if c == "#0" else "e")
            self.table.pack(fill="x", pady=3)
            ttk.Button(right, text="Save comparison as CSV", command=self.save_csv).pack(anchor="w")

        def fill_models(self):
            f = self.filter.get()
            keep = {"All models": lambda e: True, "ONNX Runtime": lambda e: e["runtime"] == "onnxruntime", "OpenVINO": lambda e: e["runtime"] == "openvino",
                    "TFLite": lambda e: e["runtime"] == "tflite", "Current stage (512 px)": lambda e: e["input_size"] == 512,
                    "INT8 only": lambda e: "INT8" in e["group"] or "all INT8" in e["label"] or e["label"].startswith(("M1", "M2", "M3"))}[f]
            self.tree.delete(*self.tree.get_children())
            self.shown = [e for e in CATALOG if keep(e)]
            for i, e in enumerate(self.shown):
                self.tree.insert("", "end", iid=str(i), text=e["name"], values=("-" if e["mAP50_95"] is None else f"{e['mAP50_95']:.3f}", "-" if e["bench_ms"] is None else f"{e['bench_ms']:.0f}"))

        def refresh_cfg(self):
            """Copy the settings into a plain dict: the worker thread reads this, never the Tk variables."""
            self.cfg = {"conf": self.conf.get(), "iou": self.iou.get(), "step": max(1, self.step.get()), "save": self.save.get(), "cam": self.cam.get()}

        def selected(self):
            return [self.shown[int(i)] for i in self.tree.selection()]

        def browse(self):
            p = filedialog.askopenfilename(title="Choose a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")], initialdir=HERE / "sample_video")
            if p:
                self.video_path = p
                self.video_label.config(text=Path(p).name)
                self.source.set("video")

        def on_select(self, _e):
            sel = self.selected()
            if sel:
                self.show_info(sel[0])
                self.wanted = sel[0]              # in webcam mode the running worker switches to this model

        def show_info(self, e):
            self.set_text(self.info, f"{e['name']}\n{e['note']}\nInput {e['input_size']} px   file {e['file_mb'] or '-'} MB\n"
                          f"Accuracy mAP50-95 {e['mAP50_95'] or '-'}  ({e['retained_pct'] or '-'}% of FP32)\nBenchmark: {e['bench_ms'] or '-'} ms/img on the test laptop")

        @staticmethod
        def set_text(box, text):
            box.config(state="normal"); box.delete("1.0", "end"); box.insert("end", text); box.config(state="disabled")

        # ---------------- running
        def start(self):
            if self.worker and self.worker.is_alive():
                return
            sel = self.selected()
            if not sel:
                messagebox.showinfo("Pick a model", "Select at least one model in the list first.")
                return
            if self.source.get() == "video" and not self.video_path:
                self.browse()
                if not self.video_path:
                    return
            self.refresh_cfg()
            self.stop.clear()
            self.rows = []
            self.table.delete(*self.table.get_children())
            self.wanted = sel[0]
            target = self.run_webcam if self.source.get() == "webcam" else self.run_videos
            self.worker = threading.Thread(target=target, args=(sel,), daemon=True)
            self.worker.start()
            self.start_btn.state(["disabled"])

        def stop_run(self):
            self.stop.set()

        def publish(self, frame, stats):
            with self.lock:
                self.latest = (frame, stats)

        def say(self, text):
            self.status_text = text

        def run_webcam(self, _sel):
            cap = cv2.VideoCapture(self.cfg["cam"], cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # a small picture keeps it fast
            if not cap.isOpened():
                self.say("Could not open the webcam. Try another number next to 'Webcam', or close other programs that use it.")
                return
            model, loaded = None, None
            try:
                while not self.stop.is_set():
                    want = self.wanted
                    if want is not loaded:
                        self.say(f"Loading {want['name']} ...")
                        if model:
                            model.close()
                        try:
                            model, loaded = engines.load_model(want, HERE), want
                        except Exception as ex:
                            self.say(f"Could not load {want['name']}: {ex}")
                            self.stop.set(); break
                        self.say(f"Running {want['name']} on the webcam. Click another model to switch.")
                    ok, frame = cap.read()
                    if not ok:
                        self.say("The webcam stopped delivering frames."); break
                    dets, t = model.detect(frame, self.cfg["conf"], self.cfg["iou"])
                    self.publish(engines.draw(frame, dets), make_stats(model, dets, t))
            finally:
                cap.release()
                if model:
                    model.close()
                self.events.put(("finished", None))

        def run_videos(self, sel):
            try:
                out_dir = HERE / "outputs"
                for i, entry in enumerate(sel, 1):
                    if self.stop.is_set():
                        break
                    self.say(f"[{i}/{len(sel)}] Loading {entry['name']} ...")
                    try:
                        model = engines.load_model(entry, HERE)
                    except Exception as ex:
                        self.say(f"Skipped {entry['name']}: {ex}")
                        continue
                    self.say(f"[{i}/{len(sel)}] Running {entry['name']} on {Path(self.video_path).name}")
                    wp = None
                    if self.cfg["save"]:
                        out_dir.mkdir(exist_ok=True)
                        wp = out_dir / (entry["id"] + ".mp4")
                    summary = run_video(model, self.video_path, self.cfg["conf"], self.cfg["iou"], self.cfg["step"], self.publish, self.stop, wp)
                    model.close()
                    if summary:
                        self.rows.append(summary)
                        self.events.put(("row", summary))
                self.say("Finished. Compare the models in the table on the right." + (f" Annotated videos are in {out_dir}." if self.cfg["save"] else ""))
            finally:
                self.events.put(("finished", None))

        def finished(self):
            self.start_btn.state(["!disabled"])

        def add_row(self, s):
            self.table.insert("", "end", text=s["model"], values=(s["frames"], f"{s['total_ms']:.0f}", f"{s['fps']:.1f}", f"{s['avg_dets']:.1f}", "-" if s["mAP"] is None else f"{s['mAP']:.3f}"))

        def save_csv(self):
            if not self.rows:
                messagebox.showinfo("Nothing to save", "Run a video through one or more models first.")
                return
            p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="model_comparison.csv")
            if p:
                with open(p, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                    w.writeheader(); w.writerows(self.rows)

        # ---------------- screen updates
        def poll(self):
            self.refresh_cfg()
            self.conf_text.config(text=f"{self.cfg['conf']:.2f}")
            self.status.config(text=self.status_text)
            while not self.events.empty():
                kind, payload = self.events.get()
                if kind == "row":
                    self.add_row(payload)
                else:
                    self.finished()
            with self.lock:
                item, self.latest = self.latest, None
            if item:
                frame, s = item
                w, h = max(self.frame_box.winfo_width(), 200), max(self.frame_box.winfo_height(), 200)
                r = min(w / frame.shape[1], h / frame.shape[0])
                img = cv2.resize(frame, (int(frame.shape[1] * r), int(frame.shape[0] * r)))
                self.photo = ImageTk.PhotoImage(Image.fromarray(img[:, :, ::-1]))
                self.video.config(image=self.photo)
                self.fps_ema = 1000 / s["total"] if not self.fps_ema else 0.9 * self.fps_ema + 0.1 * 1000 / s["total"]
                self.history = (self.history + [s["infer"]])[-100:]
                prog = f"   frame {s['frame'] + 1}/{s['total_frames']}" if s["total_frames"] else ""
                self.set_text(self.live, f"{s['model']}\nSpeed: {self.fps_ema:.1f} frames/s{prog}\nModel {s['infer']:.1f} ms   resize {s['pre']:.1f} ms   boxes {s['post']:.1f} ms\n"
                              f"Detections: {s['ndet']}   mean confidence {s['avg_conf']:.2f}")
                self.set_text(self.objs, ", ".join(f"{k}: {v}" for k, v in s["counts"].most_common()) or "nothing detected")
                self.draw_spark()
            self.after(30, self.poll)

        def draw_spark(self):
            c = self.spark
            c.delete("all")
            h = self.history
            if len(h) < 2:
                return
            W, H, top = c.winfo_width() or 440, 60, max(h) * 1.1 or 1
            pts = [(i * (W - 4) / 99 + 2, H - 4 - v / top * (H - 8)) for i, v in enumerate(h)]
            c.create_line(*[v for p in pts for v in p], fill="#1f4e9e", width=1.5)
            c.create_text(4, 2, anchor="nw", text=f"max {max(h):.0f} ms", fill="#666", font=("Segoe UI", 8))

        def quit_app(self):
            self.stop.set()
            self.destroy()

    App().mainloop()


# ----------------------------------------------------------------------------------- command line check
def selftest():
    video = HERE / "sample_video" / "demo.mp4"
    pick = [e for e in CATALOG if e["label"].startswith(("FP32 baseline | 640", "M3 per-channel | 512", "M3 | plain network | 512", "TFLite M3, per-channel | 512"))]
    print(f"{len(pick)} models on {video.name} (every 5th frame)")
    print(f"{'model':58s} {'frames':>6s} {'ms/img':>7s} {'fps':>6s} {'dets':>5s}")
    for e in pick:
        m = engines.load_model(e, HERE)
        s = run_video(m, video, 0.25, 0.45, 5, lambda f, st: None, threading.Event())
        m.close()
        print(f"{e['name'][:58]:58s} {s['frames']:6d} {s['total_ms']:7.1f} {s['fps']:6.1f} {s['avg_dets']:5.1f}")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main_window()
