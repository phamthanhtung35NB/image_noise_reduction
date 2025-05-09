import numpy as np
import cv2
import time
import os
import warnings
from datetime import datetime
import glob
import argparse

# Bỏ qua các cảnh báo
warnings.filterwarnings("ignore")
# Lưu trữ thông tin thời gian và các chỉ số
system_info = {
    'system1_time': 0.0,# Thời gian hệ thống 1
    'edge_noise_count': 0, # Số lượng nhiễu
    'flat_noise_count': 0, # Số lượng nhiễu phẳng
    'system2_time': 0.0, # Thời gian hệ thống 2
    'removed_count': 0, # Số lượng điểm bị loại bỏ
    'total_detection_time': 0.0, # Thời gian phát hiện tổng thể nhiễu
    'restore_time': 0.0 # Thời gian khôi phục
}
def fast_median_filter(img, noise_flag, window_size):
    """Áp dụng bộ lọc trung vị tối ưu hóa"""
    height, width = img.shape
    half_window = window_size // 2
    result = img.copy()  # Giữ các pixel không nhiễu nguyên vẹn
    
    # Lấy tọa độ các điểm nhiễu
    noise_points = np.where(noise_flag == 1)
    noise_coords = list(zip(noise_points[0], noise_points[1]))
    
    for i, j in noise_coords:
        # Xác định vùng cửa sổ
        i_min = max(0, i - half_window)
        i_max = min(height, i + half_window + 1)
        j_min = max(0, j - half_window)
        j_max = min(width, j + half_window + 1)
        
        # Lấy cửa sổ và mặt nạ nhiễu tương ứng
        window = img[i_min:i_max, j_min:j_max]
        window_mask = noise_flag[i_min:i_max, j_min:j_max]
        
        # Lấy các pixel không nhiễu trong cửa sổ
        good_pixels = window[window_mask == 0]
        
        # Nếu có pixel tốt, lấy giá trị trung vị
        if len(good_pixels) > 0:
            result[i, j] = np.median(good_pixels)
        else:
            # Nếu không có pixel tốt, lấy trung vị của toàn bộ cửa sổ
            result[i, j] = np.median(window)
    
    return result

def create_noise_flag(img, filtered_img, threshold):
    """Tạo ảnh cờ nhiễu tối ưu hóa"""
    diff = np.abs(img.astype(np.int32) - filtered_img.astype(np.int32))
    noise_flag = np.zeros_like(img, dtype=np.uint8)
    noise_flag[diff >= threshold] = 1
    return noise_flag

def verify_edge_noise(img, f_noise, F_edge, TN, TC, TR, G=255):
    """Xác minh các điểm nhiễu trong vùng biên - phiên bản tối ưu"""
    # Chuyển sang kiểu int32 để tránh tràn số
    img_int = img.astype(np.int32)
    F_noise = f_noise.copy()
    
    # Lấy tọa độ của tất cả điểm nhiễu trong vùng phẳng
    flat_noise_points = np.where((f_noise == 1) & (F_edge == 0))
    flat_noise_coords = list(zip(flat_noise_points[0], flat_noise_points[1]))
    
    # Lấy giá trị cường độ của các điểm nhiễu vùng phẳng
    flat_values = np.array([img_int[i, j] for i, j in flat_noise_coords])
    
    # Phân loại các điểm nhiễu vùng phẳng thành hai nhóm: dương và âm
    flat_values_pos = flat_values[flat_values >= G/2]
    flat_values_neg = flat_values[flat_values < G/2]
    
    # Lấy tọa độ của tất cả điểm nhiễu trong vùng biên cần xác minh
    edge_noise_points = np.where((f_noise == 1) & (F_edge == 1))
    edge_noise_coords = list(zip(edge_noise_points[0], edge_noise_points[1]))
    
    # Sử dụng phương pháp batch để tăng tốc
    batch_size = min(1000, len(edge_noise_coords))  # Số điểm xử lý mỗi lần
    num_batches = (len(edge_noise_coords) + batch_size - 1) // batch_size
    
    removed_count = 0
    
    # Xử lý theo batch để tránh tràn bộ nhớ
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(edge_noise_coords))
        
        if b % 10 == 0 and b > 0:
            print(f"Đã xử lý {start_idx}/{len(edge_noise_coords)} điểm, loại bỏ {removed_count} điểm")
        
        # Lấy batch hiện tại
        current_batch = edge_noise_coords[start_idx:end_idx]
        
        for i, j in current_batch:
            z_ij = img_int[i, j]
            
            # Chọn danh sách giá trị phù hợp dựa trên cường độ
            if z_ij >= G/2 and len(flat_values_pos) >= TC:
                # Tính chênh lệch với tất cả các giá trị dương
                diffs = np.abs(flat_values_pos - z_ij)
                M1 = np.sum(diffs <= TN)
                R = M1 / len(flat_values_pos)
                
                if R < TR:
                    F_noise[i, j] = 0
                    removed_count += 1
                    
            elif z_ij < G/2 and len(flat_values_neg) >= TC:
                # Tính chênh lệch với tất cả các giá trị âm
                diffs = np.abs(flat_values_neg - z_ij)
                M1 = np.sum(diffs <= TN)
                R = M1 / len(flat_values_neg)
                
                if R < TR:
                    F_noise[i, j] = 0
                    removed_count += 1
    
    return F_noise, removed_count

    
class ImpulseNoiseDetector:
    """
    Triển khai phương pháp phát hiện nhiễu xung với hai hệ thống - phiên bản tối ưu hóa
    với khả năng hiển thị kết quả từng bước
    """
    
    # Khởi tạo các tham số cho cả hai hệ thống
    def __init__(self, WE1=5, WE2=7, TE=10, 
                 WD1=7, WD2=9, TD=30, 
                 TN=10, TC=20, TR=0.8):
        """
        Khởi tạo các tham số cho cả hai hệ thống.
        """
        self.WE1 = WE1
        self.WE2 = WE2
        self.TE = TE
        self.WD1 = WD1
        self.WD2 = WD2
        self.TD = TD
        self.TN = TN
        self.TC = TC
        self.TR = TR
        # Lưu trữ các ảnh trung gian
        self.intermediate_results = {}
        
    def _median_filter(self, img, window_size):
        """Áp dụng bộ lọc trung vị với kích thước cửa sổ đã cho"""
        return cv2.medianBlur(img, window_size)
    
    def _create_noise_flag_PSM(self, img, window_size, threshold):
        """Tạo ảnh cờ nhiễu sử dụng phương pháp PSM cơ bản tối ưu hóa"""
        # Áp dụng bộ lọc trung vị
        filtered_img = self._median_filter(img, window_size)
        
        # Tính ảnh cờ nhiễu
        return create_noise_flag(img, filtered_img, threshold)
    
    def _system1(self, img):
        """
        Triển khai Hệ thống 1 - Phát hiện nhiễu dựa trên ảnh cờ biên (phiên bản tối ưu).
        """
        start_time = time.time()
        print("Bắt đầu Hệ thống 1...")
        
        # Khởi tạo các ảnh ban đầu
        height, width = img.shape
        
        # Tạo ảnh cờ nhiễu f_WD1 và f_WD2 trực tiếp - tối ưu hóa lớn
        print("Tạo ảnh cờ nhiễu từ hai bộ lọc trung vị...")
        f_WD1 = self._create_noise_flag_PSM(img, self.WD1, self.TD)
        f_WD2 = self._create_noise_flag_PSM(img, self.WD2, self.TD)
        
        # Tạo ảnh cờ biên sử dụng thuật toán Canny
        print("Phát hiện biên bằng phương pháp thay thế...")
        edges = cv2.Canny(img, 50, 150)
        
        # Dãn nở cạnh để bao phủ vùng biên tốt hơn
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Ánh xạ kết quả phát hiện biên sang ảnh cờ biên
        f_edge = np.zeros_like(img, dtype=np.uint8)
        f_edge[edges_dilated > 0] = 1
        
        # Tạo ảnh cờ nhiễu cuối cùng theo phương trình (9)
        f_noise = np.zeros_like(img, dtype=np.uint8)
        f_noise[f_edge == 1] = f_WD1[f_edge == 1]  # Vùng biên
        f_noise[f_edge == 0] = f_WD2[f_edge == 0]  # Vùng phẳng
        
        # Lưu kết quả trung gian
        self.intermediate_results['edge_detection'] = edges
        self.intermediate_results['dilated_edges'] = edges_dilated
        self.intermediate_results['f_WD1'] = f_WD1 * 255  # Nhân với 255 để dễ nhìn
        self.intermediate_results['f_WD2'] = f_WD2 * 255  # Nhân với 255 để dễ nhìn
        self.intermediate_results['f_edge'] = f_edge * 255  # Nhân với 255 để dễ nhìn
        self.intermediate_results['system1_result'] = f_noise * 255  # Nhân với 255 để dễ nhìn
        time_taken = time.time() - start_time
        print(f"Hoàn thành Hệ thống 1 trong {time_taken:.2f} giây")
        # lưu thời gian hệ thống 1
        system_info['system1_time'] = time_taken
        return f_noise, f_edge
    
    def _system2(self, img, f_noise, f_edge):
        """
        Triển khai Hệ thống 2 - Hệ thống xác minh (phiên bản tối ưu).
        """
        start_time = time.time()
        print("Bắt đầu Hệ thống 2 - Xác minh kết quả từ Hệ thống 1...")
        
        # Dãn nở ảnh cờ biên để xác định vùng biên rộng hơn
        kernel = np.ones((3, 3), dtype=np.uint8)
        F_edge = cv2.dilate(f_edge, kernel, iterations=1)
        
        # Đếm số lượng điểm cần xác minh
        num_edge_noise = np.sum((f_noise == 1) & (F_edge == 1))
        num_flat_noise = np.sum((f_noise == 1) & (F_edge == 0))

        # Lưu số lượng điểm nhiễu
        system_info['edge_noise_count'] = num_edge_noise
        system_info['flat_noise_count'] = num_flat_noise
        print(f"Có {num_edge_noise} điểm nhiễu trong vùng biên cần xác minh")
        print(f"Có {num_flat_noise} điểm nhiễu trong vùng phẳng để so sánh")
        
        # Lưu kết quả trung gian
        self.intermediate_results['dilated_edge_mask'] = F_edge * 255  # Nhân với 255 để dễ nhìn
        
        # Sử dụng hàm tối ưu hóa để xác minh điểm nhiễu
        print("Xác minh các điểm nhiễu bằng phương pháp vector hóa...")
        F_noise, removed_count = verify_edge_noise(
            img, f_noise, F_edge,
            self.TN, self.TC, self.TR
        )
        
        # Lưu kết quả trung gian
        self.intermediate_results['system2_result'] = F_noise * 255  # Nhân với 255 để dễ nhìn
        print(f"Hệ thống 2 đã loại bỏ {removed_count} điểm bị phát hiện sai")
        time_taken = time.time() - start_time
        print(f"Hoàn thành Hệ thống 2 trong {time_taken:.2f} giây")

        # lưu thời gian hệ thống 2
        system_info['system2_time'] = time_taken
        system_info['removed_count'] = removed_count

        
        return F_noise
    
    def detect(self, img):
        """
        Phát hiện nhiễu xung trong ảnh bằng cách kết hợp cả hai hệ thống.
        """
        start_time = time.time()
        
        # Kiểm tra định dạng ảnh đầu vào
        # Nếu ảnh không phải là ảnh xám, chuyển đổi sang ảnh xám
        if len(img.shape) > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        # Lưu ảnh gốc
        # self.intermediate_results['original'] = img
        
        # Áp dụng Hệ thống 1
        f_noise, f_edge = self._system1(img_gray)
        
        # Áp dụng Hệ thống 2
        F_noise = self._system2(img_gray, f_noise, f_edge)
        
        # Lưu kết quả cuối cùng
        self.intermediate_results['final_noise_flag'] = F_noise * 255  # Nhân với 255 để dễ nhìn
        
        time_taken = time.time() - start_time
        print(f"Tổng thời gian phát hiện nhiễu: {time_taken:.2f} giây")
        # lưu thời gian phát hiện nhiễu
        system_info['total_detection_time'] = time_taken
        return F_noise
    
    def restore(self, img_gray_noise, original_img=None):  
        """
        Khôi phục ảnh bị nhiễu xung bằng cách sử dụng kết quả phát hiện nhiễu.
        """
        start_time = time.time()
        print("Bắt đầu quá trình khôi phục ảnh...")
        
        
        # Lưu ảnh gốc
        self.intermediate_results['original'] = original_img
        # Lưu ảnh gốc
        # cv2.imwrite("original.png", self.intermediate_results['original'] )
        
        # Phát hiện nhiễu xung trong ảnh bằng cách kết hợp cả hai hệ thống.
        noise_flag = self.detect(img_gray_noise)
        
        # Sử dụng hàm tối ưu hóa để khôi phục ảnh
        print("Áp dụng bộ lọc trung vị tối ưu hóa để khôi phục...")
        # Khôi phục ảnh bằng bộ lọc trung vị (ảnh nhiễu - cờ nhiễu - kích thước cửa sổ)
        restored_img = fast_median_filter(img_gray_noise, noise_flag, 3)
        
        # Lưu ảnh khôi phục
        self.intermediate_results['restored'] = restored_img
        
        time_taken = time.time() - start_time
        print(f"Thời gian khôi phục ảnh: {time_taken:.2f} giây")
        # lưu thời gian khôi phục
        system_info['restore_time'] = time_taken
        return restored_img

    def write_PERFORMANCE(self, summary_img, title, width, height, row_idx, col_idx, spacing, font_size, font_thickness):
        # Tạo ảnh trắng cho phần hiển thị chỉ số
        header_font_size = 1.3  # kích thước chữ tiêu đề 
        data_font_size = 1.2  # kích thước chữ dữ liệu
        header_font_thickness = 3  # độ dày chữ tiêu đề
        data_font_thickness = 2  # độ dày chữ dữ liệu
        margin_left = 40  # khoảng cách bên trái
        
        # Tính vị trí đặt thông tin PSNR
        y_start = row_idx * (height + spacing + 55) + 10  # Thêm 20px để đẩy xuống thấp hơn
        y_end = y_start + height
        x_start = col_idx * (width + spacing)
        x_end = x_start + width

        # Vị trí bắt đầu và khoảng cách giữa các dòng
        line_y = 80  # Dòng bắt đầu
        line_spacing = 45  # Khoảng cách giữa các dòng
        
        # Tạo một ảnh trắng cho phần hiển thị chỉ số
        metrics_img = np.ones((height, width), dtype=np.uint8) * 255
        
        # Thêm tiêu đề
        cv2.putText(metrics_img, "IMAGE QUALITY:", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, header_font_size, 0, header_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        # Tính PSNR
        if 'original' in self.intermediate_results and 'noisy' in self.intermediate_results:
            psnr_noisy = cv2.PSNR(self.intermediate_results['original'], self.intermediate_results['noisy'])
            text_noisy = f"PSNR noisy: {psnr_noisy:.2f} dB"
            cv2.putText(metrics_img, text_noisy, (margin_left, line_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
            line_y += line_spacing
        
        if 'original' in self.intermediate_results and 'restored' in self.intermediate_results:
            psnr_restored = cv2.PSNR(self.intermediate_results['original'], self.intermediate_results['restored'])
            text_restored = f"PSNR restored: {psnr_restored:.2f} dB"
            cv2.putText(metrics_img, text_restored, (margin_left, line_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
            line_y += line_spacing
            
            improvement = psnr_restored - psnr_noisy
            text_improve = f"Improvement: {improvement:.2f} dB"
            cv2.putText(metrics_img, text_improve, (margin_left, line_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
            line_y += int(line_spacing * 2)
        
        # Thêm thông tin về SYSTEM PERFORMANCE
        cv2.putText(metrics_img, "SYSTEM PERFORMANCE:", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, header_font_size, 0, header_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        # System 1 info
        cv2.putText(metrics_img, f"System 1 time: {system_info['system1_time']:.2f}s", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        # Noise points info
        cv2.putText(metrics_img, f"Edge noise points: {system_info['edge_noise_count']}", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        cv2.putText(metrics_img, f"Flat noise points: {system_info['flat_noise_count']}", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        # System 2 info
        cv2.putText(metrics_img, f"System 2 time: {system_info['system2_time']:.2f}s", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        # Removed points info
        cv2.putText(metrics_img, f"Removed points: {system_info['removed_count']}", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing

        # Total times
        cv2.putText(metrics_img, f"Total detection: {system_info['total_detection_time']:.2f}s", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        line_y += line_spacing
        
        cv2.putText(metrics_img, f"Restoration time: {system_info['restore_time']:.2f}s", (margin_left, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, data_font_size, 0, data_font_thickness, cv2.LINE_AA)
        
        summary_img[y_start:y_end, x_start:x_end] = metrics_img
        
        # Thêm tiêu đề
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
        text_x = x_start + (width - text_size[0]) // 2
        text_y = y_end + 25  # Dưới ảnh
        
        cv2.putText(
            summary_img, 
            title, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_size, 
            0,  # Màu đen
            font_thickness, 
            cv2.LINE_AA
        )


    def create_summary_image(self, noisy_img=None):
        """
        Tạo ảnh tổng hợp từ các ảnh trung gian với bố cục tối ưu hơn.
        """
        # Kiểm tra xem có đầy đủ kết quả không
        required_keys = ['original', 'system1_result', 'system2_result', 'restored']
        if not all(key in self.intermediate_results for key in required_keys):
            print("Thiếu kết quả trung gian. Hãy chạy phương thức restore() trước.")
            return None
        
        # Nếu có ảnh nhiễu truyền vào, lưu lại
        if noisy_img is not None:
            if len(noisy_img.shape) > 2:
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
            self.intermediate_results['noisy'] = noisy_img
        
        # Cấu trúc hiển thị: 3 hàng x 4 cột
        image_layout = [
            # Hàng 1: Ảnh gốc và các bước xử lý chính
            [('original', 'Original'), ('noisy', 'Noisy'), ('edge_detection', 'Edge Detection'), ('dilated_edges', 'Dilated Edges')],
            # Hàng 2: Các cờ nhiễu trung gian
            [('f_WD1', 'f_WD1'), ('f_WD2', 'f_WD2'), ('f_edge', 'f_edge'), ('system1_result', 'System1 Result')],
            # Hàng 3: Kết quả hệ thống và ảnh đã khôi phục
            [('dilated_edge_mask', 'Dilated Edge Mask'), ('system2_result', 'System2 Result'), ('restored', 'Restored'), ('psnr', 'Performance Metrics')]
        ]
        
        # Lấy kích thước ảnh
        height, width = self.intermediate_results['original'].shape
        
        # Tạo ảnh tổng hợp 
        spacing = 10  # Khoảng cách giữa các ảnh
        rows = len(image_layout)
        cols = max(len(row) for row in image_layout)
        
        total_width = width * cols + spacing * (cols - 1)
        # tiêu đề có chiều cao 60 pixel
        total_height = height * rows + spacing * (rows - 1) + 55 * rows  # Thêm không gian cho tiêu đề
        
        summary_img = np.ones((total_height, total_width), dtype=np.uint8) * 240  # Nền màu xám nhạt
        
        # Đặt các ảnh vào ảnh tổng hợp
        font_size = 1.5  # Kích thước chữ tiêu đề tăng lên 1.2
        font_thickness = 2  # Độ dày chữ
        
        for row_idx, row in enumerate(image_layout):
            for col_idx, (key, title) in enumerate(row):


                # Tính vị trí đặt ảnh
                y_start = row_idx * (height + spacing + 55)
                y_end = y_start + height
                x_start = col_idx * (width + spacing)
                x_end = x_start + width
                
                # Trường hợp đặc biệt cho ô PSNR
                if key == 'psnr':
                    self.write_PERFORMANCE(
                        summary_img, 
                        title, 
                        width, height, 
                        row_idx, col_idx, 
                        spacing, font_size, font_thickness
                    )
                    continue
                
                if key is None or key not in self.intermediate_results:
                    continue
                img = self.intermediate_results[key]
                # Đảm bảo ảnh có cùng kích thước
                if img.shape != (height, width):
                    img = cv2.resize(img, (width, height))
                
                
                summary_img[y_start:y_end, x_start:x_end] = img
                
                # Thêm tiêu đề
                text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
                text_x = x_start + (width - text_size[0]) // 2
                text_y = y_end + 42  # Dưới ảnh
                
                cv2.putText(
                    summary_img, 
                    title, 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, 
                    0,  # Màu đen
                    font_thickness, 
                    cv2.LINE_AA
                )
        
        return summary_img


def add_salt_pepper_noise(img, salt_prob, pepper_prob):
    """Thêm nhiễu muối tiêu (salt-and-pepper) vào ảnh - phiên bản tối ưu hóa"""
    noisy_img = img.copy()
    height, width = img.shape
    
    # Tạo mặt nạ ngẫu nhiên
    salt_mask = np.random.random((height, width)) < salt_prob
    pepper_mask = np.random.random((height, width)) < pepper_prob
    
    # Áp dụng mặt nạ
    noisy_img[salt_mask] = 255
    noisy_img[pepper_mask] = 0
    
    return noisy_img


def process_image(input_path, output_dir, noise_level=0.3):
    """
    Xử lý ảnh đầu vào và tạo ảnh tổng hợp.
    
    Parameters:
    -----------
    input_path : str
        Đường dẫn đến ảnh đầu vào
    output_dir : str
        Thư mục để lưu ảnh đầu ra
    noise_level : float
        Mức độ nhiễu (0.0-1.0)
    """
    # Đọc ảnh
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không thể đọc ảnh từ {input_path}")
        return False
    
    # Lấy tên file
    file_name = os.path.basename(input_path)
    # Lấy tên file không có phần mở rộng
    name_without_ext = os.path.splitext(file_name)[0]
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Xử lý ảnh: {file_name} (kích thước: {img.shape})")
    # thêm anh gốc vào ảnh đầu ra
    # Tạo ảnh nhiễu
    print(f"Tạo ảnh nhiễu với {noise_level*100:.0f}% nhiễu dương và {noise_level*100:.0f}% nhiễu âm...")
    noisy_img = add_salt_pepper_noise(img, noise_level, noise_level)
    
    # Khởi tạo bộ phát hiện nhiễu
    detector = ImpulseNoiseDetector(
        WE1=5, WE2=7, TE=10,
        WD1=7, WD2=9, TD=30,
        TN=10, TC=20, TR=0.8
    )
    
    # Phục hồi ảnh
    print("Bắt đầu phát hiện và khôi phục ảnh...")
    restored_img = detector.restore(noisy_img,img)
    
    # Tạo ảnh tổng hợp
    print("Tạo ảnh tổng hợp...")
    summary_img = detector.create_summary_image(noisy_img)
    
    # Lưu ảnh tổng hợp
    output_path = os.path.join(output_dir, f"{name_without_ext}_summary.png")
    cv2.imwrite(output_path, summary_img)
    
    print(f"Đã lưu ảnh tổng hợp tại: {output_path}")
    
    # Tính PSNR nếu có thể
    # PSNR là một chỉ số đo lường chất lượng ảnh, thường được sử dụng để đánh giá hiệu suất của các thuật toán khôi phục ảnh.
    # Nó đo lường tỷ lệ tín hiệu trên nhiễu (SNR) trong ảnh.    
    # PSNR càng cao thì chất lượng ảnh càng tốt.
    # PSNR = 10 * log10((MAX^2) / MSE)
    # Trong đó:
    # - MAX là giá trị tối đa có thể của ảnh (255 cho ảnh 8-bit)
    # - MSE là sai số bình phương trung bình giữa ảnh gốc và ảnh khôi phục.
    # - MSE = (1/N) * Σ((I1 - I2)^2)
    try:
        psnr_noisy = cv2.PSNR(img, noisy_img)
        psnr_restored = cv2.PSNR(img, restored_img)
        print(f"\nSo sánh hiệu suất:")
        print(f"- PSNR ảnh nhiễu: {psnr_noisy:.2f} dB")
        print(f"- PSNR ảnh khôi phục: {psnr_restored:.2f} dB")
        print(f"- Cải thiện: {psnr_restored - psnr_noisy:.2f} dB")
    except:
        print("\nKhông thể tính PSNR.")
    
    return True


def process_directory(input_dir, output_dir, noise_level=0.3):
    """
    Xử lý tất cả ảnh trong thư mục.
    """
    # Lấy danh sách các file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục {input_dir}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Xử lý từng ảnh
    for i, file_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Đang xử lý {file_path}...")
        process_image(file_path, output_dir, noise_level)


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Phục hồi ảnh nhiễu xung')
    parser.add_argument('input', help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh')
    parser.add_argument('-o', '--output', default='outputs', help='Thư mục đầu ra (mặc định: results)')
    parser.add_argument('-n', '--noise', type=float, default=0.2, help='Mức độ nhiễu')
    
    args = parser.parse_args()
    
    # Kiểm tra đường dẫn đầu vào
    if os.path.isfile(args.input):
        # Xử lý một ảnh
        process_image(args.input, args.output, args.noise)
    elif os.path.isdir(args.input):
        # Xử lý tất cả ảnh trong thư mục
        process_directory(args.input, args.output, args.noise)
    else:
        print(f"Không tìm thấy {args.input}")


if __name__ == "__main__":
    main()