import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hệ thống 1: Phát hiện nhiễu xung cải tiến
def detect_impulse_noise_enhanced(image, edge_threshold=20, 
                                noise_threshold_edge=40, 
                                noise_threshold_flat=30, 
                                small_window=3, 
                                large_window=5):
    # 1. Tạo ảnh cờ biên (edge flag image) - sử dụng Canny thay vì Sobel
    edges = cv2.Canny(image, 50, 150)
    edge_flag = np.zeros_like(image, dtype=np.uint8)
    edge_flag[edges > 0] = 1
    
    # Áp dụng dilate để mở rộng vùng biên
    kernel = np.ones((3,3), np.uint8)
    edge_flag = cv2.dilate(edge_flag, kernel, iterations=1)
    
    # 2. Phát hiện nhiễu thông qua độ lệch với trung vị
    padded = cv2.copyMakeBorder(image, large_window//2, large_window//2, 
                              large_window//2, large_window//2, 
                              cv2.BORDER_REFLECT)
    
    noise_mask = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(large_window//2, padded.shape[0] - large_window//2):
        for j in range(large_window//2, padded.shape[1] - large_window//2):
            original_i = i - large_window//2
            original_j = j - large_window//2
            
            pixel_value = padded[i, j]
            
            # Chọn cửa sổ và ngưỡng phù hợp dựa vào ảnh cờ biên
            if edge_flag[original_i, original_j] == 1:  # Vùng biên
                window = padded[i - small_window//2:i + small_window//2 + 1, 
                              j - small_window//2:j + small_window//2 + 1]
                threshold = noise_threshold_edge
            else:  # Vùng phẳng
                window = padded[i - large_window//2:i + large_window//2 + 1, 
                              j - large_window//2:j + large_window//2 + 1]
                threshold = noise_threshold_flat
            
            # Xác định giá trị cực đại và cực tiểu
            min_val = np.min(window)
            max_val = np.max(window)
            median_val = np.median(window)
            
            # Phát hiện nhiễu - cải tiến với cả tiêu chí về giá trị cực đại/tiểu
            is_salt = (pixel_value > 200) and (abs(int(pixel_value) - median_val) > threshold)
            is_pepper = (pixel_value < 50) and (abs(int(pixel_value) - median_val) > threshold)
            
            if is_salt or is_pepper:
                noise_mask[original_i, original_j] = 255
    
    return noise_mask, edge_flag

# Hệ thống 2: Xác minh kết quả cải tiến
def verify_noise_enhanced(image, noise_mask, edge_flag, 
                        window_size=5, 
                        similarity_threshold=15, 
                        ratio_threshold=0.25):
    verified_mask = np.copy(noise_mask)
    
    padded_image = cv2.copyMakeBorder(image, window_size//2, window_size//2, 
                                    window_size//2, window_size//2, 
                                    cv2.BORDER_REFLECT)
    padded_mask = cv2.copyMakeBorder(noise_mask, window_size//2, window_size//2, 
                                   window_size//2, window_size//2, 
                                   cv2.BORDER_CONSTANT, value=0)
    
    # Xác minh chỉ các pixel được đánh dấu là nhiễu
    for i in range(window_size//2, padded_image.shape[0] - window_size//2):
        for j in range(window_size//2, padded_image.shape[1] - window_size//2):
            original_i = i - window_size//2
            original_j = j - window_size//2
            
            # Chỉ xác minh các pixel được đánh dấu là nhiễu
            if noise_mask[original_i, original_j] == 255:
                pixel_value = padded_image[i, j]
                
                window = padded_image[i - window_size//2:i + window_size//2 + 1, 
                                    j - window_size//2:j + window_size//2 + 1]
                mask_window = padded_mask[i - window_size//2:i + window_size//2 + 1, 
                                        j - window_size//2:j + window_size//2 + 1]
                
                similar_noise_pixels = 0
                total_noise_pixels = 0
                
                for m in range(window_size):
                    for n in range(window_size):
                        if mask_window[m, n] == 255:
                            total_noise_pixels += 1
                            if abs(int(window[m, n]) - int(pixel_value)) <= similarity_threshold:
                                similar_noise_pixels += 1
                
                if total_noise_pixels > 0:
                    R = similar_noise_pixels / total_noise_pixels
                    if R < ratio_threshold:
                        verified_mask[original_i, original_j] = 0
    
    return verified_mask

# Lọc nhiễu nâng cao - kết hợp nhiều phương pháp
def denoise_advanced(image, noise_mask, edge_flag):
    # Tạo bản sao cho ảnh kết quả
    result = np.copy(image)
    
    # Chuẩn bị ảnh có padding cho bộ lọc
    padded = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_REFLECT)
    
    # Áp dụng bộ lọc cho từng pixel nhiễu
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if noise_mask[i, j] == 255:  # Nếu là pixel nhiễu
                i_pad = i + 3  # Vị trí tương ứng trong ảnh đã padding
                j_pad = j + 3
                
                if edge_flag[i, j] == 1:  # Vùng biên
                    # Sử dụng cửa sổ nhỏ (3x3) với bộ lọc trung vị có trọng số
                    window = padded[i_pad-1:i_pad+2, j_pad-1:j_pad+2].astype(float)
                    
                    # Tính khoảng cách cường độ với pixel trung tâm
                    center = padded[i_pad, j_pad]
                    weight_matrix = np.exp(-0.1 * np.abs(window - center))
                    
                    # Sắp xếp giá trị và chọn trung vị có trọng số
                    values = window.flatten()
                    weights = weight_matrix.flatten()
                    sorted_indices = np.argsort(values)
                    sorted_values = values[sorted_indices]
                    sorted_weights = weights[sorted_indices]
                    cum_weights = np.cumsum(sorted_weights)
                    median_idx = np.searchsorted(cum_weights, cum_weights[-1] / 2)
                    result[i, j] = sorted_values[median_idx]
                    
                else:  # Vùng phẳng
                    # Sử dụng cửa sổ lớn hơn (5x5) với bộ lọc trung vị thông thường
                    window = padded[i_pad-2:i_pad+3, j_pad-2:j_pad+3]
                    result[i, j] = np.median(window)
    
    # Áp dụng thêm bộ lọc bilateral để giữ biên và làm mịn
    result = cv2.bilateralFilter(result, 5, 75, 75)
    
    return result

# Áp dụng hệ thống lọc nhiễu kết hợp
def full_denoising_system(image):
    # Bước 1: Phát hiện nhiễu (Hệ thống 1)
    noise_mask, edge_flag = detect_impulse_noise_enhanced(
        image, 
        edge_threshold=20, 
        noise_threshold_edge=35, 
        noise_threshold_flat=25,
        small_window=3, 
        large_window=5
    )
    
    # Bước 2: Xác minh kết quả (Hệ thống 2)
    verified_mask = verify_noise_enhanced(
        image, 
        noise_mask, 
        edge_flag, 
        window_size=5, 
        similarity_threshold=20, 
        ratio_threshold=0.2
    )
    
    # Bước 3: Lọc nhiễu nâng cao
    denoised_img = denoise_advanced(image, verified_mask, edge_flag)
    
    # Bước 4: Áp dụng thêm bộ lọc NLMeans để cải thiện kết quả
    final_result = cv2.fastNlMeansDenoising(denoised_img, None, 10, 7, 21)
    
    return noise_mask, verified_mask, edge_flag, denoised_img, final_result

# Đọc ảnh đầu vào (cần thay đổi đường dẫn)
def process_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Nếu không đọc được, tạo ảnh mẫu để kiểm tra code
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                img[i, j] = min(255, (i + j) // 2)
        cv2.line(img, (50, 50), (150, 150), 0, 2)
        # Thêm nhiễu xung
        rnd = np.random.rand(*img.shape)
        img[rnd < 0.05] = 0
        img[rnd > 0.95] = 255
    
    # Áp dụng hệ thống lọc nhiễu
    noise_mask, verified_mask, edge_flag, denoised_img, final_result = full_denoising_system(img)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(edge_flag * 255, cmap='gray')
    plt.title("Ảnh cờ biên")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(noise_mask, cmap='gray')
    plt.title("Phát hiện nhiễu (Hệ thống 1)")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(verified_mask, cmap='gray')
    plt.title("Xác minh nhiễu (Hệ thống 2)")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(denoised_img, cmap='gray')
    plt.title("Ảnh lọc nhiễu cơ bản")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(final_result, cmap='gray')
    plt.title("Ảnh kết quả cuối cùng")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img, noise_mask, verified_mask, edge_flag, denoised_img, final_result
process_image("ex2.jpg")
process_image("ex3.png")
process_image("ex4.jpg")
process_image("ex5.jpg")

'''
edge_threshold: Giảm để bắt được nhiều biên hơn
noise_threshold_edge và noise_threshold_flat: Giảm để phát hiện nhiều nhiễu hơn
similarity_threshold: Tăng để chấp nhận nhiều ứng viên nhiễu hơn
Tham số bộ lọc NLMeans (10, 7, 21): Điều chỉnh để cân bằng giữa làm mịn và giữ chi tiết
'''