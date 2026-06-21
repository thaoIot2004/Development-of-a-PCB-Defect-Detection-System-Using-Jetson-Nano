# PCB Defect Detection System using NVIDIA Jetson Nano

> Đồ án tốt nghiệp - Khoa Điện - Điện Tử, Trường ĐH Công Nghệ Kỹ Thuật TP.HCM  
> **Sinh viên:** Phan Thanh Thảo | **MSSV:** 22139062 | **GVHD:** ThS. Trương Quang Phúc

---

## 📋 Giới thiệu

Hệ thống phát hiện khuyết điểm bảng mạch in PCB tự động sử dụng mô hình học sâu YOLOv10s triển khai trên máy tính nhúng NVIDIA Jetson Nano. Giải pháp hướng đến các doanh nghiệp vừa và nhỏ, thay thế phương pháp kiểm tra thủ công bằng mắt người với độ chính xác và tốc độ vượt trội.

> 📸 **[CHÈN HÌNH: Ảnh tổng quan hệ thống thực tế]**

---

## 🎯 Tính năng

- Phát hiện **6 loại lỗi PCB** phổ biến:
  - Missing hole (thiếu lỗ khoan)
  - Mouse bite (lỗ đục biên dạng)
  - Open circuit (hở mạch)
  - Short circuit (ngắn mạch)
  - Spur (gai đồng)
  - Spurious copper (đồng thừa)
- Hỗ trợ PCB kích thước tối đa **10×10 cm**
- Thời gian xử lý **~400ms/ảnh** ở độ phân giải 640×640
- Giao diện đồ họa trực quan (Tkinter), hiển thị bounding box và thông tin lỗi
- Triển khai đóng gói hoàn toàn qua **Docker**, khởi động chỉ bằng một cú double-click

---

## 🏗️ Kiến trúc hệ thống

```
[Camera USB] → [Jetson Nano] → [Docker Container] → [YOLOv10s] → [Giao diện Tkinter]
```

Hệ thống gồm 3 khối chính:
- **Khối thu thập dữ liệu:** Camera 8MP IMX179 USB, buồng tối 3 tầng tích hợp đèn LED + tấm tản sáng
- **Khối xử lý trung tâm:** YOLOv10s chạy trong Docker container với NVIDIA GPU runtime
- **Khối hiển thị:** Giao diện Tkinter + OpenCV, hiển thị kết quả real-time

> 📸 **[CHÈN HÌNH: Sơ đồ khối tổng quát]**

---

## 🔧 Phần cứng

| Thành phần | Chi tiết |
|---|---|
| Máy tính nhúng | NVIDIA Jetson Nano (4GB) |
| Camera | 8MP Sony IMX179 USB (UVC) |
| Nguồn | 5V/4A |
| Lưu trữ | MicroSD ≥ 64GB |
| Buồng kiểm tra | In 3D 3 tầng, kích thước tổng thể 130×160×260mm |

> 📸 **[CHÈN HÌNH: Mô hình thực tế]**

---

## 🤖 Mô hình AI

- **Kiến trúc:** YOLOv10s (7.2M tham số, 21.4G FLOPs)
- **Dataset:** Ảnh PCB từ Kaggle, tiền xử lý chuyển đổi không gian màu RGB → CIELab + CLAHE
- **Phân chia dữ liệu:** Train 80% / Val 10% / Test 10%
- **Huấn luyện:** Google Colab, 50 epochs, input 640×640, batch 16

### Kết quả huấn luyện

| Metric | Giá trị |
|---|---|
| Precision | 0.9636 |
| Recall | 0.9108 |
| mAP@0.5 | 0.9551 |
| mAP@0.5:0.95 | 0.5782 |

### Độ chính xác theo từng loại lỗi

| Loại lỗi | Precision | Recall |
|---|---|---|
| Missing hole | 0.94 | 0.92 |
| Mouse bite | 0.87 | 0.83 |
| Open circuit | 0.95 | 0.94 |
| Short circuit | 0.98 | 0.97 |
| Spur | 0.86 | 0.85 |
| Spurious copper | 0.96 | 0.93 |

> 📸 **[CHÈN HÌNH: Confusion matrix normalized]**

---

## ⚙️ Cài đặt và chạy

### Yêu cầu
- NVIDIA Jetson Nano với JetPack SDK
- Docker + NVIDIA Container Runtime
- Camera USB kết nối tại `/dev/video0`

### Khởi động

```bash
# Cấp quyền thực thi
chmod +x launch_yolo.sh

# Chạy hệ thống
./launch_yolo.sh
```

Hoặc **double-click** vào file `YOLO.desktop` trên màn hình desktop.

> Script tự động khởi động Docker, cấp quyền X11 và chạy container với GPU runtime.

---

## 🖥️ Hướng dẫn sử dụng

1. **Khởi động** → Double-click biểu tượng YOLO App (~2 phút load model)
2. **Đặt PCB** vào buồng kiểm tra, đóng cửa
3. Nhấn **"Chụp & Predict"** để nhận diện, hoặc **"Chọn ảnh"** để dùng ảnh có sẵn
4. **Đọc kết quả** trên giao diện — bounding box + bảng danh sách lỗi
5. Click vào từng dòng trong bảng để **highlight** vị trí lỗi tương ứng
6. Nhấn **"Camera"** để kiểm tra bo mạch tiếp theo

> 📸 **[CHÈN HÌNH: Giao diện ứng dụng và kết quả thử nghiệm]**

---

## 📊 Kết quả so với mục tiêu

| Chỉ tiêu | Mục tiêu | Thực tế | Đánh giá |
|---|---|---|---|
| Thời gian xử lý/ảnh | < 1000ms | ~400ms | ✅ Vượt mục tiêu |
| Kích thước PCB tối đa | 10×10cm | 10×10cm | ✅ Đạt |
| Số loại lỗi | 6 | 6 | ✅ Đạt |
| mAP@0.5 | > 0.90 | 0.9551 | ✅ Vượt mục tiêu |
| Chi phí phần cứng | Phù hợp SME | ~100 USD | ✅ Đạt |

---

## 🛠️ Công nghệ sử dụng

![Python](https://img.shields.io/badge/Python-3.x-blue)
![YOLOv10](https://img.shields.io/badge/YOLOv10-s-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Jetson](https://img.shields.io/badge/NVIDIA-Jetson_Nano-76B900)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

- **AI/ML:** Ultralytics YOLOv10, PyTorch, CUDA, cuDNN, TensorRT
- **Xử lý ảnh:** OpenCV, Pillow, CIELab color space, CLAHE
- **Giao diện:** Python Tkinter + ttk
- **Triển khai:** Docker, NVIDIA Container Runtime, X11

---


---

## 🔮 Hướng phát triển

- [ ] Nâng cấp lên **Jetson Orin Nano / Xavier NX** để tăng tốc độ xử lý
- [ ] Chuyển đổi model sang **TensorRT INT8/FP16** (giảm thêm 30–50% latency)
- [ ] Tích hợp **SAHI** (phân mảnh ảnh) để hỗ trợ PCB kích thước lớn
- [ ] Xây dựng **cơ sở dữ liệu** lưu trữ lịch sử kiểm định và báo cáo thống kê
- [ ] Phát triển hướng **tự động hóa hoàn toàn** (băng tải + cảm biến quang)
- [ ] Tích hợp **API/MQTT** kết nối với hệ thống MES nhà máy

---

## 📄 License

MIT License

---

*Đồ án tốt nghiệp - Ngành Hệ thống nhúng và IoT - 06/2026*
