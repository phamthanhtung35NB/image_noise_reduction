import numpy as np
import cv2
from scipy import ndimage

class ImpulseNoiseDetector:
    """
    Triển khai phương pháp phát hiện nhiễu xung với hai hệ thống:
    - Hệ thống 1: Phát hiện nhiễu dựa trên ảnh cờ biên (edge flag image)
    - Hệ thống 2: Hệ thống xác minh lại kết quả từ Hệ thống 1
    """
    
    def __init__(self, WE1=5, WE2=7, TE=10, 
                 WD1=7, WD2=9, TD=30, 
                 TN=10, TC=20, TR=0.8):
        """
        Khởi tạo các tham số cho cả hai hệ thống.
        
        Parameters:
        -----------
        WE1, WE2 : int
            Kích thước cửa sổ nhỏ và lớn cho việc tạo ảnh cờ biên
        TE : int 
            Ngưỡng cho việc xác định biên
        WD1, WD2 : int
            Kích thước cửa sổ nhỏ và lớn cho bộ lọc trung vị
        TD : int
            Ngưỡng cho việc phát hiện nhiễu
        TN : int
            Ngưỡng cho phạm vi biến đổi của nhiễu xung
        TC : int
            Số lượng tối thiểu điểm cần so sánh trong Hệ thống 2
        TR : float
            Ngưỡng tỷ lệ cho việc xác minh nhiễu trong Hệ thống 2 (0 < TR ≤ 1)
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
        """Áp dụng bộ lọc trung vị với kích thước cửa sổ đã cho."""
        return cv2.medianBlur(img, window_size)
    
    def _create_noise_flag_PSM(self, img, window_size, threshold):
        """
        Tạo ảnh cờ nhiễu sử dụng phương pháp PSM cơ bản.
        
        Parameters:
        -----------
        img : ndarray
            Ảnh đầu vào (thang xám)
        window_size : int
            Kích thước cửa sổ cho bộ lọc trung vị
        threshold : int
            Ngưỡng để xác định pixel nhiễu
            
        Returns:
        --------
        noise_flag : ndarray
            Ảnh cờ nhiễu (1: nhiễu, 0: không phải nhiễu)
        """
        # Áp dụng bộ lọc trung vị
        filtered_img = self._median_filter(img, window_size)
        
        # Tính độ chênh lệch giữa ảnh gốc và ảnh đã lọc
        diff = np.abs(img.astype(np.int32) - filtered_img.astype(np.int32))
        
        # Tạo ảnh cờ nhiễu
        noise_flag = np.zeros_like(img, dtype=np.uint8)
        noise_flag[diff >= threshold] = 1
        
        return noise_flag
    
    def _system1(self, img):
        """
        Triển khai Hệ thống 1 - Phát hiện nhiễu dựa trên ảnh cờ biên.
        
        Parameters:
        -----------
        img : ndarray
            Ảnh đầu vào (thang xám)
            
        Returns:
        --------
        f_noise : ndarray
            Ảnh cờ nhiễu (1: nhiễu, 0: không phải nhiễu)
        f_edge : ndarray
            Ảnh cờ biên (1: biên, 0: không phải biên)
        """
        # Khởi tạo các ảnh ban đầu
        height, width = img.shape
        f_edge = np.zeros((height, width), dtype=np.uint8)  # Ảnh cờ biên
        y = img.copy()  # Ảnh thang xám
        
        # Khởi tạo ảnh cờ nhiễu g từ bộ lọc trung vị lớn
        g = self._create_noise_flag_PSM(img, self.WD2, self.TD)  
        
        print(f"Bắt đầu Hệ thống 1 với {np.sum(g)} điểm nhiễu cần xử lý...")
        
        # Lặp đến khi tất cả các điểm nhiễu đã được xử lý
        iter_count = 0
        max_iterations = 50  # Đặt giới hạn số vòng lặp để tránh vòng lặp vô hạn
        
        while np.sum(g) > 0 and iter_count < max_iterations:
            print(f"Vòng lặp {iter_count+1}: Còn {np.sum(g)} điểm nhiễu cần xử lý")
            
            # Lưu trữ ảnh cường độ trước đó để kiểm tra sự thay đổi
            y_prev = y.copy()
            
            # Tạo mặt nạ cho các pixel chưa xử lý
            unprocessed_mask = (g == 1)
            unprocessed_indices = np.where(unprocessed_mask)
            
            # Xử lý từng điểm ảnh chưa được xử lý (cách tiếp cận vector hóa)
            for idx in range(len(unprocessed_indices[0])):
                i, j = unprocessed_indices[0][idx], unprocessed_indices[1][idx]
                
                # Tính giá trị trung vị cho cửa sổ nhỏ (WE1)
                i_min_small = max(0, i - self.WE1 // 2)
                i_max_small = min(height, i + self.WE1 // 2 + 1)
                j_min_small = max(0, j - self.WE1 // 2)
                j_max_small = min(width, j + self.WE1 // 2 + 1)
                
                window_small = y[i_min_small:i_max_small, j_min_small:j_max_small]
                mask_small = g[i_min_small:i_max_small, j_min_small:j_max_small] == 0
                good_pixels_small = window_small[mask_small]
                M_WE1 = len(good_pixels_small)
                
                # Tính giá trị trung vị cho cửa sổ lớn (WE2)
                i_min_large = max(0, i - self.WE2 // 2)
                i_max_large = min(height, i + self.WE2 // 2 + 1)
                j_min_large = max(0, j - self.WE2 // 2)
                j_max_large = min(width, j + self.WE2 // 2 + 1)
                
                window_large = y[i_min_large:i_max_large, j_min_large:j_max_large]
                mask_large = g[i_min_large:i_max_large, j_min_large:j_max_large] == 0
                good_pixels_large = window_large[mask_large]
                M_WE2 = len(good_pixels_large)
                
                # Tính trung vị nếu có đủ pixel tốt
                m_WE1 = np.median(good_pixels_small) if M_WE1 > 0 else y[i, j]
                m_WE2 = np.median(good_pixels_large) if M_WE2 > 0 else y[i, j]
                
                # Cập nhật ảnh cờ biên và ảnh thang xám
                if M_WE1 > 0 and M_WE2 > 0:
                    if abs(float(m_WE1) - float(m_WE2)) >= self.TE:
                        f_edge[i, j] = 1  # Đánh dấu là biên
                        y[i, j] = m_WE1   # Cập nhật theo phương trình (6)
                    else:
                        y[i, j] = m_WE2   # Vùng phẳng
                
                # Cập nhật ảnh cờ nhiễu g theo phương trình (7)
                if y[i, j] != y_prev[i, j]:
                    g[i, j] = 0  # Đã xử lý
            
            # Kiểm tra xem có thay đổi nào không để tránh vòng lặp vô hạn
            if np.array_equal(y, y_prev):
                print(f"Cảnh báo: Không có thay đổi sau vòng lặp {iter_count+1}")
                # Nếu không có thay đổi nào, tăng ngưỡng để đảm bảo tiến trình
                if iter_count % 5 == 0:  # Mỗi 5 vòng lặp không có tiến triển
                    print("Điều chỉnh ngưỡng để đẩy nhanh tiến trình")
                    # Đánh dấu tất cả các điểm nhiễu còn lại là được xử lý
                    g[g == 1] = 0
            
            iter_count += 1
        
        if iter_count >= max_iterations:
            print(f"Cảnh báo: Đã đạt đến số vòng lặp tối đa ({max_iterations})")
            print(f"Còn {np.sum(g)} điểm nhiễu chưa được xử lý")
            # Đánh dấu tất cả các điểm nhiễu còn lại là không phải biên
            g[g == 1] = 0
        
        # Tạo ảnh cờ nhiễu f_WD1 và f_WD2 từ hai bộ lọc trung vị khác nhau
        f_WD1 = self._create_noise_flag_PSM(img, self.WD1, self.TD)
        f_WD2 = self._create_noise_flag_PSM(img, self.WD2, self.TD)
        
        # Tạo ảnh cờ nhiễu cuối cùng theo phương trình (9)
        f_noise = np.zeros_like(img, dtype=np.uint8)
        f_noise[f_edge == 1] = f_WD1[f_edge == 1]  # Vùng biên
        f_noise[f_edge == 0] = f_WD2[f_edge == 0]  # Vùng phẳng
        
        return f_noise, f_edge
    
    def _system2(self, img, f_noise, f_edge):
        """
        Triển khai Hệ thống 2 - Hệ thống xác minh.
        
        Parameters:
        -----------
        img : ndarray
            Ảnh đầu vào (thang xám)
        f_noise : ndarray
            Ảnh cờ nhiễu từ Hệ thống 1
        f_edge : ndarray
            Ảnh cờ biên từ Hệ thống 1
            
        Returns:
        --------
        F_noise : ndarray
            Ảnh cờ nhiễu đã được xác minh
        """
        print("Bắt đầu Hệ thống 2 - Xác minh kết quả từ Hệ thống 1...")
        
        # Dãn nở ảnh cờ biên để xác định vùng biên rộng hơn
        kernel = np.ones((3, 3), dtype=np.uint8)
        F_edge = cv2.dilate(f_edge, kernel, iterations=1)
        
        # Tạo bản sao ảnh cờ nhiễu để cập nhật
        F_noise = f_noise.copy()
        
        # Lấy danh sách các điểm cần xác minh (nhiễu trong vùng biên)
        edge_noise_points = np.where((f_noise == 1) & (F_edge == 1))
        edge_noise_coords = list(zip(edge_noise_points[0], edge_noise_points[1]))
        
        print(f"Có {len(edge_noise_coords)} điểm nhiễu trong vùng biên cần xác minh")
        
        # Giá trị cường độ cực đại
        G = 255
        
        # Lấy toàn bộ các điểm nhiễu trong vùng phẳng để so sánh
        flat_noise_points = np.where((f_noise == 1) & (F_edge == 0))
        flat_noise_coords = list(zip(flat_noise_points[0], flat_noise_points[1]))
        
        print(f"Có {len(flat_noise_coords)} điểm nhiễu trong vùng phẳng để so sánh")
        
        # Chuyển ảnh sang kiểu int32 để tránh tràn số
        img_int = img.astype(np.int32)
        
        # Đếm số điểm nhiễu được loại bỏ
        removed_count = 0
        
        # Với mỗi điểm nhiễu trong vùng biên, xác minh lại
        for idx, (i, j) in enumerate(edge_noise_coords):
            if idx % 100 == 0 and idx > 0:
                print(f"Đã xác minh {idx}/{len(edge_noise_coords)} điểm, loại bỏ {removed_count} điểm")
            
            z_ij = img_int[i, j]  # Giá trị cường độ của điểm mục tiêu
            
            # Lọc các điểm có cùng loại cường độ (dương hoặc âm)
            if z_ij >= G/2:  # Nhiễu dương
                valid_coords = [(s, t) for s, t in flat_noise_coords if img_int[s, t] >= G/2]
            else:  # Nhiễu âm
                valid_coords = [(s, t) for s, t in flat_noise_coords if img_int[s, t] < G/2]
            
            # Nếu không đủ điểm để so sánh, giữ nguyên kết quả
            if len(valid_coords) < self.TC:
                continue
            
            # Tính chênh lệch cường độ và đếm số điểm thỏa mãn
            diffs = [abs(img_int[s, t] - z_ij) for s, t in valid_coords]
            M1 = sum(1 for diff in diffs if diff <= self.TN)
            
            # Tính tỷ lệ R
            R = M1 / len(valid_coords)
            
            # Quyết định cuối cùng theo phương trình (12)
            if R < self.TR:
                F_noise[i, j] = 0  # Không phải nhiễu xung
                removed_count += 1
        
        print(f"Hệ thống 2 đã loại bỏ {removed_count} điểm bị phát hiện sai")
        return F_noise
    
    def detect(self, img):
        """
        Phát hiện nhiễu xung trong ảnh bằng cách kết hợp cả hai hệ thống.
        
        Parameters:
        -----------
        img : ndarray
            Ảnh đầu vào (thang xám)
            
        Returns:
        --------
        F_noise : ndarray
            Ảnh cờ nhiễu cuối cùng (1: nhiễu, 0: không phải nhiễu)
        """
        # Kiểm tra định dạng ảnh đầu vào
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng Hệ thống 1
        f_noise, f_edge = self._system1(img)
        
        # Áp dụng Hệ thống 2
        F_noise = self._system2(img, f_noise, f_edge)
        
        return F_noise
    
# ằng cách sử dụng kết quả phát hiện nhiễu.
    def restore(self, img):
        """
        Parameters:
        -----------
        img : ndarray
            Ảnh đầu vào (thang xám) bị nhiễu
            
        Returns:
        --------
        restored_img : ndarray
            Ảnh đã được khôi phục
        """
        # Phát hiện nhiễu
        noise_flag = self.detect(img)
        
        # Tạo bản sao ảnh để khôi phục
        restored_img = img.copy()
        
        # Khôi phục từng điểm ảnh bị đánh dấu là nhiễu
        height, width = img.shape
        
        for i in range(height):
            for j in range(width):
                if noise_flag[i, j] == 1:
                    # Lấy cửa sổ 3x3 xung quanh điểm ảnh
                    i_min = max(0, i - 1)
                    i_max = min(height, i + 2)
                    j_min = max(0, j - 1)
                    j_max = min(width, j + 2)
                    
                    window = img[i_min:i_max, j_min:j_max]
                    window_flag = noise_flag[i_min:i_max, j_min:j_max]
                    
                    # Lọc những điểm không phải nhiễu trong cửa sổ
                    good_pixels = window[window_flag == 0]
                    
                    # Nếu có điểm tốt, thay thế bằng trung vị của các điểm tốt
                    if len(good_pixels) > 0:
                        restored_img[i, j] = np.median(good_pixels)
                    else:
                        # Nếu không có điểm tốt trong cửa sổ, sử dụng bộ lọc trung vị thông thường
                        restored_img[i, j] = self._median_filter(img, 3)[i, j]
        
        return restored_img


# Hàm tạo ảnh nhiễu xung (salt-and-pepper) để kiểm tra
def add_salt_pepper_noise(img, salt_prob, pepper_prob):
    """
    Thêm nhiễu muối tiêu (salt-and-pepper) vào ảnh.
    
    Parameters:
    -----------
    img : ndarray
        Ảnh đầu vào
    salt_prob : float
        Xác suất xuất hiện nhiễu dương (salt - trắng)
    pepper_prob : float
        Xác suất xuất hiện nhiễu âm (pepper - đen)
        
    Returns:
    --------
    noisy_img : ndarray
        Ảnh đã thêm nhiễu
    """
    noisy_img = img.copy()
    
    # Thêm muối (salt - nhiễu trắng)
    salt_mask = np.random.random(img.shape) < salt_prob
    noisy_img[salt_mask] = 255
    
    # Thêm tiêu (pepper - nhiễu đen)
    pepper_mask = np.random.random(img.shape) < pepper_prob
    noisy_img[pepper_mask] = 0
    
    return noisy_img


# Hàm chính để demo phương pháp
def main():
    # Đọc ảnh mẫu - thay đổi đường dẫn nếu cần
    try:
        img = cv2.imread('ex11.jpg', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Không thể đọc ảnh từ đường dẫn.")
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {e}")
        print("Tạo ảnh mẫu để demo.")
        # Tạo ảnh mẫu nếu không đọc được
        img = np.ones((256, 256), dtype=np.uint8) * 128
        # Tạo một số cạnh và vùng phẳng
        img[50:200, 50:200] = 200
        img[75:175, 75:175] = 100
        img[100:150, 100:150] = 150
    
    print("Tạo ảnh nhiễu với 30% nhiễu dương và 30% nhiễu âm...")
    # Tạo ảnh nhiễu với 30% nhiễu dương và 30% nhiễu âm (tổng 60% nhiễu)
    noisy_img = add_salt_pepper_noise(img, 0.3, 0.3)
    
    print("Khởi tạo bộ phát hiện nhiễu với các tham số từ bài báo...")
    # Khởi tạo bộ phát hiện nhiễu với các tham số từ bài báo
    detector = ImpulseNoiseDetector(
        WE1=5, WE2=7, TE=10,
        WD1=7, WD2=9, TD=30,
        TN=10, TC=20, TR=0.8
    )
    
    print("Phát hiện nhiễu...")
    # Phát hiện nhiễu
    noise_flag = detector.detect(noisy_img)
    
    print("Khôi phục ảnh...")
    # Khôi phục ảnh
    restored_img = detector.restore(noisy_img)
    
    # Lưu kết quả thay vì hiển thị (để tránh vấn đề với GUI)
    print("Lưu kết quả...")
    cv2.imwrite('original.png', img)
    cv2.imwrite('noisy.png', noisy_img)
    cv2.imwrite('noise_flag.png', noise_flag * 255)  # Nhân với 255 để dễ nhìn
    cv2.imwrite('restored.png', restored_img)
    
    print("Hoàn thành! Các ảnh đã được lưu: original.png, noisy.png, noise_flag.png, restored.png")
    
    # Nếu muốn hiển thị kết quả (có thể gây lỗi nếu không có GUI)
    try:
        cv2.imshow('Original Image', img)
        cv2.imshow('Noisy Image (60% noise)', noisy_img)
        cv2.imshow('Noise Flag', noise_flag * 255)  # Nhân với 255 để dễ nhìn
        cv2.imshow('Restored Image', restored_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Không thể hiển thị ảnh: {e}")
        print("Các ảnh đã được lưu thành công, bạn có thể xem chúng trong thư mục hiện tại.")


if __name__ == "__main__":
    main()