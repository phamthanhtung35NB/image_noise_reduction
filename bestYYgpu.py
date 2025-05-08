import numpy as np
import cv2
from scipy import ndimage
import time
import os
import warnings

# Bỏ qua các cảnh báo
warnings.filterwarnings("ignore")

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
    """
    
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
        
        print(f"Hoàn thành Hệ thống 1 trong {time.time() - start_time:.2f} giây")
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
        
        print(f"Có {num_edge_noise} điểm nhiễu trong vùng biên cần xác minh")
        print(f"Có {num_flat_noise} điểm nhiễu trong vùng phẳng để so sánh")
        
        # Sử dụng hàm tối ưu hóa để xác minh điểm nhiễu
        print("Xác minh các điểm nhiễu bằng phương pháp vector hóa...")
        F_noise, removed_count = verify_edge_noise(
            img, f_noise, F_edge,
            self.TN, self.TC, self.TR
        )
        
        print(f"Hệ thống 2 đã loại bỏ {removed_count} điểm bị phát hiện sai")
        print(f"Hoàn thành Hệ thống 2 trong {time.time() - start_time:.2f} giây")
        
        return F_noise
    
    def detect(self, img):
        """
        Phát hiện nhiễu xung trong ảnh bằng cách kết hợp cả hai hệ thống.
        """
        start_time = time.time()
        
        # Kiểm tra định dạng ảnh đầu vào
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng Hệ thống 1
        f_noise, f_edge = self._system1(img)
        
        # Áp dụng Hệ thống 2
        F_noise = self._system2(img, f_noise, f_edge)
        
        print(f"Tổng thời gian phát hiện nhiễu: {time.time() - start_time:.2f} giây")
        return F_noise
    
    def restore(self, img):
        """
        Khôi phục ảnh bị nhiễu xung bằng cách sử dụng kết quả phát hiện nhiễu.
        """
        start_time = time.time()
        print("Bắt đầu quá trình khôi phục ảnh...")
        
        # Chuyển ảnh sang thang xám nếu cần
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện nhiễu
        noise_flag = self.detect(img)
        
        # Sử dụng hàm tối ưu hóa để khôi phục ảnh
        print("Áp dụng bộ lọc trung vị tối ưu hóa để khôi phục...")
        restored_img = fast_median_filter(img, noise_flag, 3)
        
        print(f"Thời gian khôi phục ảnh: {time.time() - start_time:.2f} giây")
        return restored_img


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


def main():
    """Hàm chính để demo phương pháp tối ưu hóa"""
    # Đọc ảnh mẫu hoặc tạo ảnh mẫu nếu không có
    try:
        img = cv2.imread('ex1111mau.jpg', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Không thể đọc ảnh từ đường dẫn.")
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {e}")
        print("Tạo ảnh mẫu để demo.")
        # Tạo ảnh mẫu
        img = np.ones((512, 512), dtype=np.uint8) * 128
        # Tạo một số cạnh và vùng phẳng
        img[100:400, 100:400] = 200
        img[150:350, 150:350] = 100
        img[200:300, 200:300] = 150
    
    print(f"Kích thước ảnh: {img.shape}")
    
    # Tạo thư mục để lưu kết quả
    os.makedirs('results', exist_ok=True)
    
    # Lưu ảnh gốc
    cv2.imwrite('results/original.png', img)
    
    print("Tạo ảnh nhiễu với 30% nhiễu dương và 30% nhiễu âm...")
    # Đo thời gian tạo nhiễu
    start_time = time.time()
    noisy_img = add_salt_pepper_noise(img, 0.3, 0.3)
    print(f"Thời gian tạo nhiễu: {time.time() - start_time:.2f} giây")
    
    # Lưu ảnh nhiễu
    cv2.imwrite('results/noisy.png', noisy_img)
    
    print("Khởi tạo bộ phát hiện nhiễu với các tham số từ bài báo...")
    # Khởi tạo bộ phát hiện nhiễu
    detector = ImpulseNoiseDetector(
        WE1=5, WE2=7, TE=10,
        WD1=7, WD2=9, TD=30,
        TN=10, TC=20, TR=0.8
    )
    
    # Phát hiện và khôi phục ảnh
    print("\n===== Bắt đầu phát hiện và khôi phục ảnh =====")
    restored_img = detector.restore(noisy_img)
    
    # Lưu kết quả
    cv2.imwrite('results/restored.png', restored_img)
    
    print("\nHoàn thành! Các ảnh đã được lưu trong thư mục 'results':")
    print("- results/original.png: Ảnh gốc")
    print("- results/noisy.png: Ảnh nhiễu")
    print("- results/restored.png: Ảnh đã khôi phục")
    
    # Hiển thị thông tin hiệu suất
    try:
        psnr_noisy = cv2.PSNR(img, noisy_img)
        psnr_restored = cv2.PSNR(img, restored_img)
        print(f"\nSo sánh hiệu suất:")
        print(f"- PSNR ảnh nhiễu: {psnr_noisy:.2f} dB")
        print(f"- PSNR ảnh khôi phục: {psnr_restored:.2f} dB")
        print(f"- Cải thiện: {psnr_restored - psnr_noisy:.2f} dB")
    except:
        print("\nKhông thể tính PSNR.")


if __name__ == "__main__":
    main()