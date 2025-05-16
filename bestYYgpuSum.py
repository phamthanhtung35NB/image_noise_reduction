import numpy as np 
import cv2
import time 
import os # Dùng để tạo thư mục và lưu ảnh
import warnings # Dùng để bỏ qua các cảnh báo
from datetime import datetime 
import glob # Dùng đệ quy tìm kiếm file
import argparse

#py -3.13 bestYYgpuSum.py inputs/1.jpg
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
    """
    Khôi phục ảnh bằng bộ lọc trung vị có chọn lọc - chỉ áp dụng lọc cho những điểm được phát hiện là nhiễu.
    
    Tham số:
    - img: Ảnh đầu vào cần được khôi phục
    - noise_flag: Mặt nạ nhị phân đánh dấu vị trí của các điểm nhiễu (1: nhiễu, 0: không nhiễu)
    - window_size: Kích thước cửa sổ bộ lọc trung vị (thường là 3, 5, 7...)
    
    Kết quả:
    - Ảnh đã được khôi phục
    """
    height, width = img.shape # Lấy kích thước ảnh (số hàng và số cột)
    half_window = window_size // 2 # Tính bán kính cửa sổ (ví dụ: window_size=3 thì half_window=1)
    result = img.copy()  # Sao chép ảnh gốc - các pixel không nhiễu sẽ giữ nguyên giá trị
    
    # Xác định vị trí các điểm nhiễu từ mặt nạ noise_flag
    # np.where trả về tọa độ của tất cả các điểm có giá trị = 1 trong mặt nạ nhiễu
    noise_points = np.where(noise_flag == 1)
    # Ghép cặp tọa độ x, y thành danh sách các điểm (i, j)
    noise_coords = list(zip(noise_points[0], noise_points[1]))
    
    # Xử lý từng điểm nhiễu một
    for i, j in noise_coords:
        # BƯỚC 2.1: Xác định vùng lân cận (cửa sổ) xung quanh điểm nhiễu
        # Đảm bảo không vượt quá biên của ảnh
        i_min = max(0, i - half_window)  # Giới hạn hàng trên
        i_max = min(height, i + half_window + 1)  # Giới hạn hàng dưới
        j_min = max(0, j - half_window)  # Giới hạn cột trái
        j_max = min(width, j + half_window + 1)  # Giới hạn cột phải
        
        # BƯỚC 2.2: Trích xuất vùng cửa sổ và mặt nạ nhiễu tương ứng
        # Lấy giá trị của các pixel trong vùng cửa sổ
        window = img[i_min:i_max, j_min:j_max]
        # Lấy giá trị mặt nạ nhiễu tương ứng với vùng cửa sổ
        window_mask = noise_flag[i_min:i_max, j_min:j_max]
        
        # BƯỚC 2.3: Chỉ sử dụng các pixel không nhiễu trong cửa sổ để tính trung vị
        # Lọc ra các pixel không nhiễu (có giá trị mặt nạ = 0)
        good_pixels = window[window_mask == 0]
        
        # BƯỚC 2.4: Tính giá trị trung vị và gán cho điểm nhiễu
        if len(good_pixels) > 0:
            # Nếu có ít nhất một pixel không nhiễu trong cửa sổ,
            # sử dụng giá trị trung vị của các pixel không nhiễu
            result[i, j] = np.median(good_pixels)
        else:
            # Nếu tất cả các pixel trong cửa sổ đều là nhiễu,
            # sử dụng giá trị trung vị của toàn bộ cửa sổ (kể cả các điểm nhiễu)
            # Đây là trường hợp xấu nhất nhưng ít khi xảy ra
            result[i, j] = np.median(window)
    
    # Trả về ảnh đã được khôi phục
    return result


def create_noise_flag(img, filtered_img, threshold):
    """
    Tạo ảnh cờ nhiễu bằng cách so sánh ảnh gốc với ảnh đã lọc.
    
    Tham số:
    - img: Ảnh gốc (có thể chứa nhiễu)
    - filtered_img: Ảnh đã được lọc (thường là kết quả từ bộ lọc trung vị)
    - threshold: Ngưỡng để xác định một pixel có phải là nhiễu hay không
    
    Kết quả trả về:
    - Mặt nạ nhị phân (1: điểm nhiễu, 0: điểm không nhiễu)
    """
    # Tính sự khác biệt giữa ảnh gốc và ảnh được lọc
    # Chuyển đổi sang kiểu int32 để tránh tràn số khi tính toán hiệu
    diff = np.abs(img.astype(np.int32) - filtered_img.astype(np.int32))
    
    # Tạo mặt nạ nhị phân ban đầu (toàn bộ là 0 - không phải nhiễu)
    noise_flag = np.zeros_like(img, dtype=np.uint8)
    
    # Đánh dấu các điểm có sự khác biệt lớn hơn hoặc bằng ngưỡng là điểm nhiễu
    # Ý tưởng: Nếu giá trị pixel trước và sau khi lọc khác nhau nhiều,
    # thì pixel đó có khả năng cao là nhiễu
    noise_flag[diff >= threshold] = 1
    
    # Trả về mặt nạ nhị phân đánh dấu các điểm nhiễu
    return noise_flag


def verify_edge_noise(img, f_noise, F_edge, TN, TC, TR, G=255):
    """
    Hàm xác minh các điểm nhiễu trong vùng biên bằng cách so sánh với các điểm nhiễu trong vùng phẳng.
    Mục đích: Loại bỏ các điểm biên bị phát hiện nhầm thành nhiễu.
    
    Tham số:
    - img: ảnh đầu vào (ảnh xám)
    - f_noise: ảnh cờ nhiễu từ Hệ thống 1 (1: điểm nhiễu, 0: điểm không nhiễu)
    - F_edge: ảnh cờ biên đã được dãn nở (1: điểm thuộc biên, 0: điểm thuộc vùng phẳng)
    - TN: ngưỡng hiệu cường độ để so sánh giữa các điểm nhiễu
    - TC: số lượng tối thiểu điểm nhiễu trong vùng phẳng cần có để thực hiện so sánh
    - TR: tỷ lệ ngưỡng (nếu tỷ lệ điểm nhiễu tương đồng thấp hơn ngưỡng này, điểm sẽ bị loại bỏ)
    - G: giá trị cường độ tối đa của ảnh (mặc định là 255 cho ảnh 8-bit)
    
    Kết quả trả về:
    - F_noise: ảnh cờ nhiễu sau khi đã loại bỏ các điểm phát hiện sai
    - removed_count: số lượng điểm bị loại bỏ
    """
    # Chuyển ảnh sang kiểu int32 để tránh tràn số khi tính toán hiệu các pixel
    img_int = img.astype(np.int32)
    # Sao chép ảnh cờ nhiễu để không làm thay đổi đầu vào
    F_noise = f_noise.copy()
    
    # Lấy danh sách các điểm nhiễu trong vùng phẳng làm tham chiếu
    # Điều kiện: (f_noise == 1) là điểm nhiễu VÀ (F_edge == 0) là điểm thuộc vùng phẳng
    flat_noise_points = np.where((f_noise == 1) & (F_edge == 0))
    # Tạo danh sách các tọa độ (i,j) từ kết quả np.where
    flat_noise_coords = list(zip(flat_noise_points[0], flat_noise_points[1]))
    
    # Lấy giá trị cường độ của tất cả các điểm nhiễu trong vùng phẳng
    # Giá trị này sẽ được dùng để so sánh với các điểm nhiễu trong vùng biên
    flat_values = np.array([img_int[i, j] for i, j in flat_noise_coords])
    
    # Phân loại các điểm nhiễu vùng phẳng thành hai nhóm dựa trên cường độ
    # - Nhóm dương (nhiễu muối): cường độ >= G/2 (thường là 128)
    # - Nhóm âm (nhiễu tiêu): cường độ < G/2
    flat_values_pos = flat_values[flat_values >= G/2]  # Điểm nhiễu dương (muối)
    flat_values_neg = flat_values[flat_values < G/2]   # Điểm nhiễu âm (tiêu)
    
    # Lấy danh sách các điểm nhiễu trong vùng biên cần xác minh
    # Điều kiện: (f_noise == 1) là điểm nhiễu VÀ (F_edge == 1) là điểm thuộc vùng biên
    edge_noise_points = np.where((f_noise == 1) & (F_edge == 1))
    # Tạo danh sách các tọa độ (i,j) từ kết quả np.where
    edge_noise_coords = list(zip(edge_noise_points[0], edge_noise_points[1]))
    
    # Chuẩn bị xử lý theo batch để tối ưu hóa bộ nhớ và hiệu suất
    # Kích thước batch là 1000 điểm hoặc ít hơn nếu tổng số điểm ít hơn 1000
    batch_size = min(1000, len(edge_noise_coords))  
    # Tính số batch cần xử lý
    num_batches = (len(edge_noise_coords) + batch_size - 1) // batch_size
    
    # Biến đếm số điểm bị loại bỏ sau khi xác minh
    removed_count = 0
    
    # Xử lý từng batch các điểm nhiễu trong vùng biên
    for b in range(num_batches):
        # Tính chỉ số bắt đầu và kết thúc của batch hiện tại
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(edge_noise_coords))
        
        # Hiển thị thông tin tiến trình mỗi 10 batch
        if b % 10 == 0 and b > 0:
            print(f"Đã xử lý {start_idx}/{len(edge_noise_coords)} điểm, loại bỏ {removed_count} điểm")
        
        # Lấy danh sách các điểm cần xử lý trong batch hiện tại
        current_batch = edge_noise_coords[start_idx:end_idx]
        
        # Xét từng điểm nhiễu trong batch hiện tại
        for i, j in current_batch:
            # Lấy giá trị cường độ của điểm hiện tại
            z_ij = img_int[i, j]
            
            # So sánh điểm nhiễu vùng biên với các điểm nhiễu vùng phẳng cùng loại
            # Xét trường hợp điểm nhiễu dương (nhiễu muối)
            if z_ij >= G/2 and len(flat_values_pos) >= TC:
                # Tính chênh lệch cường độ với tất cả các điểm nhiễu dương trong vùng phẳng
                diffs = np.abs(flat_values_pos - z_ij)
                # Đếm số điểm có chênh lệch nhỏ hơn hoặc bằng ngưỡng TN
                M1 = np.sum(diffs <= TN)
                # Tính tỷ lệ điểm tương đồng
                R = M1 / len(flat_values_pos)
                
                # Nếu tỷ lệ điểm tương đồng thấp hơn ngưỡng TR, loại bỏ điểm khỏi danh sách nhiễu
                # Ý tưởng: Nếu điểm này không giống với đa số điểm nhiễu trong vùng phẳng, có thể đây không phải là nhiễu
                if R < TR:
                    F_noise[i, j] = 0  # Đánh dấu không phải nhiễu
                    removed_count += 1  # Tăng biến đếm
            
            # Xét trường hợp điểm nhiễu âm (nhiễu tiêu)       
            elif z_ij < G/2 and len(flat_values_neg) >= TC:
                # Tính chênh lệch cường độ với tất cả các điểm nhiễu âm trong vùng phẳng
                diffs = np.abs(flat_values_neg - z_ij)
                # Đếm số điểm có chênh lệch nhỏ hơn hoặc bằng ngưỡng TN
                M1 = np.sum(diffs <= TN)
                # Tính tỷ lệ điểm tương đồng
                R = M1 / len(flat_values_neg)
                
                # Nếu tỷ lệ điểm tương đồng thấp hơn ngưỡng TR, loại bỏ điểm khỏi danh sách nhiễu
                if R < TR:
                    F_noise[i, j] = 0  # Đánh dấu không phải nhiễu
                    removed_count += 1  # Tăng biến đếm
    
    # Trả về kết quả sau khi xác minh
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
        Triển khai Hệ thống 1 - Phát hiện nhiễu dựa trên ảnh cờ biên.
        Hệ thống này phân biệt giữa các vùng biên và vùng phẳng, 
        sau đó áp dụng các cách phát hiện nhiễu khác nhau cho mỗi vùng.
        """
        # Ghi lại thời điểm bắt đầu để đo thời gian thực hiện
        start_time = time.time()
        print("Bắt đầu Hệ thống 1...")
        
        # Lấy kích thước của ảnh đầu vào
        height, width = img.shape
        
        # Tạo các ảnh cờ nhiễu cho vùng biên và vùng phẳng
        print("Tạo ảnh cờ nhiễu từ hai bộ lọc trung vị...")
        # f_WD1: ảnh cờ nhiễu cho vùng biên - sử dụng cửa sổ lớn hơn (WD1)
        # Hàm _create_noise_flag_PSM tạo cờ nhiễu dựa trên Peak-Signal-to-Median (PSM)
        f_WD1 = self._create_noise_flag_PSM(img, self.WD1, self.TD)
        
        # f_WD2: ảnh cờ nhiễu cho vùng phẳng - sử dụng cửa sổ nhỏ hơn (WD2)
        # Phát hiện nhiễu trong vùng phẳng yêu cầu cửa sổ nhỏ hơn để tăng độ nhạy
        f_WD2 = self._create_noise_flag_PSM(img, self.WD2, self.TD)
        
        # Phát hiện biên trong ảnh
        print("Phát hiện biên bằng phương pháp thay thế...")
        # Sử dụng thuật toán Canny để phát hiện biên - một thuật toán phát hiện biên phổ biến
        # Các tham số 50, 150 là ngưỡng thấp và cao cho thuật toán Canny
        edges = cv2.Canny(img, 50, 150)
        
        # Dãn nở biên để bao phủ vùng biên tốt hơn
        # Tạo kernel 3x3 cho phép toán dãn nở
        kernel = np.ones((3, 3), np.uint8)
        # Dãn nở biên để đảm bảo không bỏ sót các điểm biên
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Tạo ảnh cờ biên (edge flag)
        # Ảnh cờ biên là ảnh nhị phân: 1 cho điểm biên, 0 cho điểm không phải biên
        f_edge = np.zeros_like(img, dtype=np.uint8)
        # Đặt giá trị 1 cho tất cả các điểm trong ảnh đã dãn nở
        f_edge[edges_dilated > 0] = 1
        
        # Kết hợp hai ảnh cờ nhiễu dựa trên ảnh cờ biên
        # Tạo ảnh cờ nhiễu cuối cùng theo phương trình (9) của thuật toán
        f_noise = np.zeros_like(img, dtype=np.uint8)
        # Đối với vùng biên (f_edge == 1), sử dụng kết quả từ f_WD1
        f_noise[f_edge == 1] = f_WD1[f_edge == 1]
        # Đối với vùng phẳng (f_edge == 0), sử dụng kết quả từ f_WD2
        f_noise[f_edge == 0] = f_WD2[f_edge == 0]
        
        # Lưu các kết quả trung gian để hiển thị và đánh giá
        # Lưu các ảnh với hệ số nhân 255 để dễ nhìn (chuyển từ nhị phân [0,1] sang [0,255])
        self.intermediate_results['edge_detection'] = edges  # Kết quả phát hiện biên gốc
        self.intermediate_results['dilated_edges'] = edges_dilated  # Kết quả sau khi dãn nở biên
        self.intermediate_results['f_WD1'] = f_WD1 * 255  # Cờ nhiễu cho vùng biên
        self.intermediate_results['f_WD2'] = f_WD2 * 255  # Cờ nhiễu cho vùng phẳng
        self.intermediate_results['f_edge'] = f_edge * 255  # Cờ biên
        self.intermediate_results['system1_result'] = f_noise * 255  # Kết quả cuối cùng của Hệ thống 1
        
        # Tính thời gian thực hiện và hiển thị
        time_taken = time.time() - start_time
        print(f"Hoàn thành Hệ thống 1 trong {time_taken:.2f} giây")
        
        # Cập nhật thông tin thời gian cho mục đích đánh giá hiệu suất
        system_info['system1_time'] = time_taken
        
        # Trả về hai kết quả: ảnh cờ nhiễu và ảnh cờ biên để sử dụng trong Hệ thống 2
        return f_noise, f_edge
    

    def _system2(self, img, f_noise, f_edge):
        # Bắt đầu đếm thời gian cho Hệ thống 2
        start_time = time.time()
        print("Bắt đầu Hệ thống 2 - Xác minh kết quả từ Hệ thống 1...")
        
        # Tạo kernel 3x3 đặc (toàn 1) để thực hiện phép dãn nở
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        # Dãn nở ảnh cờ biên để tạo vùng biên rộng hơn
        # Mục đích: Tạo một vùng đệm xung quanh biên để đảm bảo không bỏ sót điểm biên
        F_edge = cv2.dilate(f_edge, kernel, iterations=1)
        
        # Đếm số lượng điểm nhiễu trong vùng biên (điểm được đánh dấu là nhiễu và nằm trong vùng biên)
        # Sử dụng phép AND logic: (f_noise == 1) & (F_edge == 1)
        num_edge_noise = np.sum((f_noise == 1) & (F_edge == 1))
        
        # Đếm số lượng điểm nhiễu trong vùng phẳng (điểm được đánh dấu là nhiễu nhưng không thuộc vùng biên)
        # Sử dụng phép AND logic: (f_noise == 1) & (F_edge == 0)
        num_flat_noise = np.sum((f_noise == 1) & (F_edge == 0))
    
        # Lưu thông tin số lượng nhiễu vào biến toàn cục để hiển thị trong báo cáo
        system_info['edge_noise_count'] = num_edge_noise
        system_info['flat_noise_count'] = num_flat_noise
        
        # Hiển thị thông tin về số lượng nhiễu để theo dõi
        print(f"Có {num_edge_noise} điểm nhiễu trong vùng biên cần xác minh")
        print(f"Có {num_flat_noise} điểm nhiễu trong vùng phẳng để so sánh")
        
        # Lưu kết quả trung gian vào danh sách các ảnh trung gian để hiển thị
        # Nhân với 255 để chuyển từ ảnh nhị phân [0,1] sang dạng [0,255] dễ nhìn
        self.intermediate_results['dilated_edge_mask'] = F_edge * 255
        
        # Thực hiện xác minh nhiễu bằng hàm verify_edge_noise (đã được tối ưu hóa)
        print("Xác minh các điểm nhiễu bằng phương pháp vector hóa...")
        
        # Gọi hàm verify_edge_noise với các tham số:
        # - img: ảnh gốc
        # - f_noise: ảnh cờ nhiễu từ Hệ thống 1
        # - F_edge: ảnh cờ biên đã dãn nở
        # - self.TN: ngưỡng hiệu cường độ để xác định nhiễu
        # - self.TC: số lượng tối thiểu điểm nhiễu trong vùng phẳng để so sánh
        # - self.TR: tỷ lệ điểm cần thỏa mãn để xác minh
        F_noise, removed_count = verify_edge_noise(
            img, f_noise, F_edge,
            self.TN, self.TC, self.TR
        )
        
        # Lưu kết quả cuối cùng của Hệ thống 2 để hiển thị
        self.intermediate_results['system2_result'] = F_noise * 255
        
        # Hiển thị số lượng điểm nhiễu đã bị loại bỏ sau khi xác minh
        print(f"Hệ thống 2 đã loại bỏ {removed_count} điểm bị phát hiện sai")
        
        # Tính toán và hiển thị thời gian thực hiện Hệ thống 2
        time_taken = time.time() - start_time
        print(f"Hoàn thành Hệ thống 2 trong {time_taken:.2f} giây")
    
        # Lưu thời gian thực hiện và số điểm loại bỏ vào thông tin hệ thống
        system_info['system2_time'] = time_taken
        system_info['removed_count'] = removed_count
    
        # Trả về ảnh cờ nhiễu sau khi đã được xác minh
        return F_noise
    
    """
    Phát hiện nhiễu xung trong ảnh bằng cách kết hợp cả hai hệ thống.
    """
    def detect(self, img):
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
    

        #Khôi phục ảnh bị nhiễu xung
    def restore(self, img_gray_noise, original_img=None):  
        """
        Phương thức khôi phục ảnh bị nhiễu xung bằng cách phát hiện và loại bỏ nhiễu.
        
        Tham số:
        - img_gray_noise: Ảnh xám bị nhiễu cần khôi phục
        - original_img: Ảnh gốc (không nhiễu) để tính PSNR và đánh giá hiệu suất (tùy chọn)
        
        Kết quả trả về:
        - Ảnh đã được khôi phục
        """
        # Bắt đầu đo thời gian khôi phục
        start_time = time.time()
        print("Bắt đầu quá trình khôi phục ảnh...")
        
        # Lưu ảnh gốc (không nhiễu) vào danh sách các ảnh trung gian
        # Ảnh này sẽ được sử dụng để tính PSNR và hiển thị trong bảng so sánh
        self.intermediate_results['original'] = original_img
        # cv2.imwrite("original.png", self.intermediate_results['original'] )
        
        # Phát hiện nhiễu xung trong ảnh
        # Gọi phương thức detect() kết hợp cả hai hệ thống để xác định vị trí các điểm nhiễu
        # Kết quả là một mặt nạ nhị phân (0: không phải nhiễu, 1: nhiễu)
        noise_flag = self.detect(img_gray_noise)
        
        # Khôi phục ảnh bằng bộ lọc trung vị có chọn lọc
        # Chỉ áp dụng bộ lọc trung vị cho các điểm đã được xác định là nhiễu
        # Các điểm không phải nhiễu giữ nguyên giá trị ban đầu
        print("Áp dụng bộ lọc trung vị tối ưu hóa để khôi phục...")
        # Tham số:
        # - img_gray_noise: ảnh nhiễu đầu vào
        # - noise_flag: mặt nạ đánh dấu các điểm nhiễu (kết quả từ hệ thống phát hiện)
        # - 3: kích thước cửa sổ bộ lọc trung vị (3x3)
        restored_img = fast_median_filter(img_gray_noise, noise_flag, 3)
        
        # Lưu kết quả ảnh đã khôi phục để hiển thị và đánh giá
        self.intermediate_results['restored'] = restored_img
        
        # Tính toán và lưu thời gian thực hiện để đánh giá hiệu suất
        time_taken = time.time() - start_time
        print(f"Thời gian khôi phục ảnh: {time_taken:.2f} giây")
        
        # Lưu thời gian khôi phục vào biến toàn cục system_info
        # Thông tin này sẽ được hiển thị trong bảng so sánh hiệu suất
        system_info['restore_time'] = time_taken
        
        # Trả về ảnh đã khôi phục
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

#Tạo ảnh tổng hợp từ các ảnh trung gian với bố cục tối ưu hơn.
    def create_summary_image(self, noisy_img=None):
        
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
            # Hàng 1: Ảnh gốc và các bước xử lý ban đầu
            [('original', 'Original'),   # Ảnh gốc không nhiễu
             ('noisy', 'Noisy'),  # Ảnh có nhiễu muối tiêu
             ('edge_detection', 'Edge Detection'),  # Kết quả phát hiện biên từ Canny
             ('dilated_edges', 'Dilated Edges')],  # Biên sau khi được dãn nở để tạo vùng biên
        
            # Hàng 2: Các cờ nhiễu trung gian từ Hệ thống 1
            [('f_WD1', 'f_WD1'),  # Cờ nhiễu cho vùng biên (dùng cửa sổ lớn WD1)
             ('f_WD2', 'f_WD2'),     # Cờ nhiễu cho vùng phẳng (dùng cửa sổ nhỏ WD2)
             ('f_edge', 'f_edge'),   # Ảnh cờ biên nhị phân (1=biên, 0=phẳng)
             ('system1_result', 'System1 Result')],  # Kết quả cuối cùng từ Hệ thống 1 (kết hợp f_WD1 và f_WD2)
        

            # Hàng 3: Kết quả cuối cùng và hiệu suất hệ thống
            [('dilated_edge_mask', 'Dilated Edge Mask'),  # Mặt nạ biên đã dãn nở thêm để dùng trong Hệ thống 2
             ('system2_result', 'System2 Result'),      # Kết quả sau khi lọc bỏ nhiễu sai từ Hệ thống 2
             ('restored', 'Restored'),                 # Ảnh đã được khôi phục hoàn chỉnh
             ('psnr', 'Performance Metrics')]       # Panel hiển thị các chỉ số hiệu suất của thuật toán
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
    


#Thêm nhiễu muối tiêu (salt-and-pepper) vào ảnh - phiên bản tối ưu hóa
#salt_prob: xác suất cho nhiễu muối (giá trị 255)
#pepper_prob: xác suất cho nhiễu tiêu (giá trị 0)
def add_salt_pepper_noise(img, salt_prob, pepper_prob):
    noisy_img = img.copy()
    height, width = img.shape
    
    # Tạo mặt nạ ngẫu nhiên
    #salt_mask: mặt nạ cho nhiễu muối (giá trị 255)
    salt_mask = np.random.random((height, width)) < salt_prob

    #pepper_mask: mặt nạ cho nhiễu tiêu (giá trị 0)
    pepper_mask = np.random.random((height, width)) < pepper_prob
    
    # Áp dụng mặt nạ
    noisy_img[salt_mask] = 255
    noisy_img[pepper_mask] = 0
    
    return noisy_img


'''
Xử lý ảnh đầu vào và tạo ảnh tổng hợp.

Parameters:
-----------
input_path : str
    Đường dẫn đến ảnh đầu vào
output_dir : str
    Thư mục để lưu ảnh đầu ra
noise_level : float
    Mức độ nhiễu (0.0-1.0)
'''
def process_image(input_path, output_dir, noise_level=0.2):
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

    # Tạo ảnh nhiễu
    print(f"Tạo ảnh nhiễu với {noise_level*100:.0f}% nhiễu dương và {noise_level*100:.0f}% nhiễu âm...")
    noisy_img = add_salt_pepper_noise(img, noise_level, noise_level)
    
    # Khởi tạo bộ phát hiện nhiễu
    '''
    Phát hiện biên:
    WE1=5: Kích thước cửa sổ 5×5 cho việc phát hiện biên trong vùng phức tạp
    WE2=7: Kích thước cửa sổ 7×7 cho việc phát hiện biên trong vùng có chi tiết mịn hơn
    TE=10: Ngưỡng hiệu cường độ để xác định một điểm ảnh có phải là điểm biên hay không

    Phát hiện nhiễu:
    WD1=7: Kích thước cửa sổ 7×7 cho việc phát hiện nhiễu ở vùng biên (lớn hơn để bao quát được nhiều thông tin biên)
    WD2=9: Kích thước cửa sổ 9×9 cho việc phát hiện nhiễu ở vùng phẳng (lớn hơn để lấy được thông tin trung bình vùng rộng)
    TD=30: Ngưỡng sai biệt giữa pixel gốc và pixel sau khi lọc trung vị để xác định một điểm có phải nhiễu hay không

    Xác minh và lọc nhiễu:
    TN=10: Ngưỡng hiệu cường độ để so sánh giữa điểm nhiễu ở vùng biên và vùng phẳng
    TC=20: Số lượng tối thiểu điểm nhiễu trong vùng phẳng cần có để thực hiện so sánh với một điểm nhiễu ở vùng biên
    TR=0.8: Tỷ lệ ngưỡng (80%) - nếu tỷ lệ điểm nhiễu tương đồng thấp hơn ngưỡng này, điểm sẽ bị loại bỏ khỏi danh sách nhiễu
    '''
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

# Xử lý tất cả ảnh trong thư mục
def process_directory(input_dir, output_dir, noise_level=0.2):
    # Lấy danh sách các file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        # Tìm tất cả file ảnh trong thư mục với các định dạng khác nhau
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        # image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục {input_dir}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Xử lý từng ảnh
    for i, file_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Đang xử lý {file_path}...")
        process_image(file_path, output_dir, noise_level)


def main():
    # parse : tham số đầu vào
    parser = argparse.ArgumentParser(description='Phục hồi ảnh nhiễu xung')
    parser.add_argument('input', help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh')
    parser.add_argument('-o', '--output', default='outputs', help='Thư mục đầu ra')
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

'''
image_layout = [
    # Hàng 1: Ảnh gốc và các bước xử lý ban đầu
    [('original', 'Original'),   # Ảnh gốc không nhiễu
    ('noisy', 'Noisy'),  # Ảnh có nhiễu muối tiêu
    ('edge_detection', 'Edge Detection'),  # Kết quả phát hiện biên từ Canny
    ('dilated_edges', 'Dilated Edges')],  # Biên sau khi được dãn nở để tạo vùng biên

    # Hàng 2: Các cờ nhiễu trung gian từ Hệ thống 1
    [('f_WD1', 'f_WD1'),  # Cờ nhiễu cho vùng biên (dùng cửa sổ lớn WD1)
    ('f_WD2', 'f_WD2'),     # Cờ nhiễu cho vùng phẳng (dùng cửa sổ nhỏ WD2)
    ('f_edge', 'f_edge'),   # Ảnh cờ biên nhị phân (1=biên, 0=phẳng)
    ('system1_result', 'System1 Result')],  # Kết quả cuối cùng từ Hệ thống 1 (kết hợp f_WD1 và f_WD2)


    # Hàng 3: Kết quả cuối cùng và hiệu suất hệ thống
    [('dilated_edge_mask', 'Dilated Edge Mask'),  # Mặt nạ biên đã dãn nở thêm để dùng trong Hệ thống 2
    ('system2_result', 'System2 Result'),      # Kết quả sau khi lọc bỏ nhiễu sai từ Hệ thống 2
    ('restored', 'Restored'),                 # Ảnh đã được khôi phục hoàn chỉnh
    ('psnr', 'Performance Metrics')]       # Panel hiển thị các chỉ số hiệu suất của thuật toán
]
'''