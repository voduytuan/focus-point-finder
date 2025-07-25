# Focus Point Finder

🚀 **Focus Point Finder** là một API mã nguồn mở giúp bạn xác định điểm trọng tâm (focus point) trong ảnh. API sử dụng machine learning (InsightFace) để nhận diện khuôn mặt và OpenCV để phân tích độ nổi bật (saliency) nếu không có khuôn mặt nào được phát hiện.

Mục tiêu: cung cấp một điểm gợi ý `{x, y}` giúp bạn crop ảnh đẹp hơn, tránh mất nội dung chính.

---

## ✨ Tính năng

- ✅ Tự động xác định khuôn mặt trong ảnh (nhiều khuôn mặt → chọn tối đa 3 khuôn mặt lớn nhất).
- ✅ Tính trung bình tọa độ các khuôn mặt → làm điểm focus.
- ✅ Fallback sang Saliency Detection nếu không có khuôn mặt.
- ✅ Trả về API JSON hoặc ảnh minh họa (debug).
- ✅ Docker-ready để dễ deploy.

---

## 🧪 API Endpoint

| Phương thức | Endpoint                  | Mô tả |
|------------|----------------------------|-------|
| GET        | `/focus-point`             | Trả về `{x, y}` focus point từ ảnh |
| GET        | `/debug/face-boxes`        | Trả về danh sách bounding box khuôn mặt |
| GET        | `/debug/image-with-boxes`  | Trả về ảnh JPG với khung đỏ quanh mặt |
| GET        | `/debug/image-with-focus`  | Trả về ảnh JPG với chấm đỏ tại focus point |

---

## 🚀 Cách sử dụng nhanh với Docker

```bash
git clone https://github.com/your-name/focus-point-finder.git
cd focus-point-finder

docker build -t focus-point-finder .
docker run -p 8000:8000 focus-point-finder
```

Gọi thử:
```
http://localhost:8000/focus-point?image_url=https://your-image-url.jpg
```

---

## 📝 License

MIT © 2024. Đóng góp tự do. Nếu bạn dùng dự án này trong sản phẩm thương mại, hãy ghi nguồn nếu có thể ❤️
