# image_noise_reduction

# Bài báo: An Accurate Noise Detector for Image Restoration

## 1. Giới thiệu
Bài báo trình bày giải pháp phát hiện nhiễu xung (impulsive noise) chính xác nhằm cải thiện độ phục hồi ảnh so với các phương pháp truyền thống như lọc trung vị (median filter) hay Progressive Switching Median (PSM). Phương pháp đề xuất gồm hai giai đoạn:

- Phát hiện nhiễu dựa trên ảnh cờ biên (edge flag image) và hai bộ lọc trung vị kích thước khác nhau.
- Xác minh các pixel nghi nhiễu ở khu vực biên bằng phân tích độ giống nhau về cường độ.

## 2. Mô hình nhiễu xung
Nhiễu xung được mô hình hóa dưới dạng các pixel có giá trị cực đại (muối) hoặc cực tiểu (tiêu), xuất hiện với xác suất nhất định. Sự khác biệt giữa các pixel nhiễu cùng loại được giả định nhỏ hơn ngưỡng Tₙ.

## 3. Bộ lọc PSM và nhược điểm
PSM sử dụng bộ phát hiện nhiễu và lọc trung vị có chọn lọc. Tuy nhiên:
- **Bỏ sót nhiễu**: xảy ra trong vùng phẳng khi pixel nhiễu không khác biệt đủ so với giá trị median.
- **Nhận nhầm biên**: xảy ra ở vùng biên khi pixel tốt bị nhầm là nhiễu do khác biệt lớn với median.

## 4. Phương pháp phát hiện nhiễu đề xuất
### 4.1 Hệ thống 1: Dựa trên ảnh cờ biên
- Dùng hai bộ lọc median (cửa sổ nhỏ và lớn) để xử lý riêng vùng biên và vùng phẳng.
- Cải thiện khả năng phát hiện và giảm nhận nhầm pixel.

### 4.2 Hệ thống 2: Xác minh kết quả
- Kiểm tra lại các pixel nghi nhiễu bằng cách so sánh mức độ tương đồng với lân cận.
- Giảm đáng kể nhận nhầm trong vùng biên.

## 5. Kết quả thực nghiệm
- Ảnh phục hồi có chất lượng vượt trội so với median lặp và PSM.
- Tỷ lệ phát hiện nhiễu chính xác cao hơn rõ rệt.
- Đặc biệt hiệu quả khi tỉ lệ nhiễu cao.

## 6. Kết luận
Phương pháp đề xuất phát hiện nhiễu hiệu quả và chính xác, giúp cải thiện đáng kể chất lượng phục hồi ảnh. Khắc phục hoàn toàn các nhược điểm chính của PSM.

---

**Tác giả**: Keiko Kondo et al.  
**Tài liệu gốc**: An Accurate Noise Detector for Image Restoration (IEEE)  
