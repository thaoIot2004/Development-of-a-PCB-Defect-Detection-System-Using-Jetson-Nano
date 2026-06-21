import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Preprocessing: BGR → CIE LAB + CLAHE
# Phải khớp với pipeline lúc build dataset:
#   clipLimit=2.0, tileGridSize=(8, 8)
# ─────────────────────────────────────────────
def preprocess_to_lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.merge([l_clahe, a, b])

class YoloApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Detection App")
        self.geometry("900x750")
        self.configure(bg="#0f172a")

        # ===== MODEL =====
        self.model = YOLO("model.pt")
        # Gán tên nhãn thủ công vì model train thiếu metadata tên class
        self.class_names = {
            0: 'Missing_hole',
            1: 'Mouse_bite',
            2: 'Open_circuit',
            3: 'Short',
            4: 'Spur',
            5: 'Spurious_copper',
        }

        # ===== STYLE =====
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#0f172a')
        style.configure('TLabel', background='#0f172a', foreground='#f8fafc', font=('Helvetica', 10))
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), foreground='#f8fafc')
        style.configure('Status.TLabel', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'), borderwidth=0, focuscolor='none', relief='flat')
        style.map('TButton',
                  background=[('active', '#2563eb'), ('disabled', '#475569')],
                  foreground=[('disabled', '#94a3b8')])
        style.configure('Capture.TButton', background='#3b82f6', foreground='white')
        style.configure('File.TButton', background='#64748b', foreground='white')
        style.configure('Back.TButton', background='#f97316', foreground='white')
        style.map('Back.TButton', background=[('active', '#ea580c'), ('disabled', '#9a3412')])
        style.configure('Treeview', background='#1e293b', foreground='#f8fafc', fieldbackground='#1e293b', font=('Helvetica', 9))
        style.map('Treeview', background=[('selected', '#3b82f6')])
        style.configure('Treeview.Heading', background='#334155', foreground='#f8fafc', font=('Helvetica', 10, 'bold'))

        # ===== UI LAYOUT =====
        self.grid_rowconfigure(1, weight=1)   # image frame
        self.grid_rowconfigure(4, weight=1)   # tree frame
        self.grid_columnconfigure(0, weight=1)

        # ----- Title -----
        title = ttk.Label(self, text=" YOLO Object Detection", style='Title.TLabel')
        title.grid(row=0, column=0, pady=(15, 5))

        # ----- Image display -----
        self.img_label = tk.Label(self, bg="#1e293b", bd=2, relief='flat', highlightthickness=0)
        self.img_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # ----- Button frame -----
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, pady=10)

        self.btn_capture = ttk.Button(btn_frame, text="Chụp & Predict", command=self.on_capture, style='Capture.TButton')
        self.btn_capture.grid(row=0, column=0, padx=5)

        self.btn_file = ttk.Button(btn_frame, text="Chọn ảnh", command=self.on_select_file, style='File.TButton')
        self.btn_file.grid(row=0, column=1, padx=5)

        self.btn_back = ttk.Button(btn_frame, text="Camera", command=self.on_back_to_camera, style='Back.TButton', state='disabled')
        self.btn_back.grid(row=0, column=2, padx=5)

        # ----- Status label -----
        self.lbl_status = ttk.Label(self, text="Sẵn sàng", style='Status.TLabel', foreground="#4ade80")
        self.lbl_status.grid(row=3, column=0, pady=5)

        # ----- Treeview for objects -----
        tree_frame = ttk.Frame(self)
        tree_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, columns=('label', 'conf', 'bbox'), show='tree headings', height=8)
        self.tree.heading('#0', text='STT')
        self.tree.heading('label', text='Nhãn')
        self.tree.heading('conf', text='Độ tin cậy')
        self.tree.heading('bbox', text='Tọa độ (x1,y1,x2,y2)')
        self.tree.column('#0', width=50, anchor='center')
        self.tree.column('label', width=120, anchor='center')
        self.tree.column('conf', width=100, anchor='center')
        self.tree.column('bbox', width=250, anchor='center')
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind('<<TreeviewSelect>>', self.on_select_object)

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)

        # ===== CAMERA =====
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.current_boxes = []
        self.original_result_frame = None
        self.is_showing_result = False

        self._update_preview()

    # =========================
    # Preview camera
    # =========================
    def _update_preview(self):
        if not self.is_showing_result:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self._show_frame(frame)
        self.after(30, self._update_preview)

    # =========================
    # Quay lại camera
    # =========================
    def on_back_to_camera(self):
        self.is_showing_result = False
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.current_boxes = []
        self.original_result_frame = None
        self.btn_back.config(state='disabled')
        self.lbl_status.config(text="Sẵn sàng", foreground="#4ade80")

    # =========================
    # Capture từ camera
    # =========================
    def on_capture(self):
        if self.current_frame is None:
            return
        self._set_busy("Đang predict từ camera...")
        threading.Thread(target=self._predict, args=(self.current_frame.copy(),), daemon=True).start()

    # =========================
    # Chọn file
    # =========================
    def on_select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            self.lbl_status.config(text="Lỗi đọc ảnh", foreground="#f87171")
            return
        self._set_busy("Đang predict từ file...")
        threading.Thread(target=self._predict, args=(frame,), daemon=True).start()

    # =========================
    # Predict
    # =========================
    def _predict(self, frame):
        try:
            # Giữ frame BGR gốc để vẽ bbox và hiển thị màu thật cho người dùng
            frame_orig = frame.copy()
            # Chuyển sang CIE LAB + CLAHE — đúng format model đã được train
            frame_lab = preprocess_to_lab(frame)
            results = self.model(frame_lab, imgsz=640, device='cpu', conf=0.2)
            res = results[0]
            boxes = res.boxes
            names = self.class_names  # tên nhãn gán thủ công

            objects = []
            # Vẽ lên ảnh gốc BGR — không vẽ lên ảnh LAB
            annotated = frame_orig.copy()

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{names[cls_id]} {conf:.2f}"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 120, 255), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 120, 255), -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    objects.append({
                        "label": names[cls_id],
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2)
                    })

            self.after(0, lambda: self._show_result(annotated, objects))

        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text=f"Lỗi: {str(e)}", foreground="#f87171"))
            self.after(0, self._set_ready)

    # =========================
    # Hiển thị kết quả
    # =========================
    def _show_result(self, frame, objects):
        self.is_showing_result = True
        self._show_frame(frame)

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.current_boxes = objects
        self.original_result_frame = frame.copy()

        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj["bbox"]
            self.tree.insert('', 'end', iid=str(i), text=str(i+1),
                             values=(obj['label'], f"{obj['conf']:.2f}", f"({x1},{y1},{x2},{y2})"))

        self.lbl_status.config(text=f"Phát hiện {len(objects)} đối tượng", foreground="#4ade80")
        self.btn_back.config(state='normal')
        self._set_ready()

    # =========================
    # Click highlight object
    # =========================
    def on_select_object(self, event):
        selected = self.tree.selection()
        if not selected or self.original_result_frame is None:
            return
        idx = int(selected[0])
        obj = self.current_boxes[idx]

        frame = self.original_result_frame.copy()
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{obj['label']} {obj['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self._show_frame(frame)

    # =========================
    # Hiển thị ảnh
    # =========================
    def _show_frame(self, frame, size=(800, 450)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img

    # =========================
    # UI state
    # =========================
    def _set_busy(self, text):
        self.btn_capture.config(state='disabled')
        self.btn_file.config(state='disabled')
        self.btn_back.config(state='disabled')
        self.lbl_status.config(text=text, foreground="#fbbf24")

    def _set_ready(self):
        self.btn_capture.config(state='normal')
        self.btn_file.config(state='normal')
        # back button state được quản lý riêng trong _show_result và on_back_to_camera

    # =========================
    # Close app
    # =========================
    def on_close(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = YoloApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
