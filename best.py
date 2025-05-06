import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm thêm nhiễu xung (salt & pepper noise) vào ảnh
def add_salt_pepper_noise(image, prob):
    noisy = np.copy(image)
    # Tạo ma trận ngẫu nhiên cùng kích thước với ảnh gốc
    rnd = np.random.rand(*image.shape)
    
    # Gán pixel đen (0) với xác suất prob/2
    noisy[rnd < (prob / 2)] = 0  # "pepper" - nhiễu đen
    
    # Gán pixel trắng (255) với xác suất prob/2
    noisy[rnd > 1 - (prob / 2)] = 255  # "salt" - nhiễu trắng
    
    return noisy

# Hệ thống 1: Phát hiện dựa trên ảnh cờ biên
def detect_impulse_noise_system1(image, edge_threshold=30, noise_threshold_edge=40, noise_threshold_flat=20, small_window=3, large_window=5):
    # Tạo ảnh cờ biên (edge flag image)
    # Sử dụng bộ lọc Sobel để phát hiện biên
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Tạo ảnh cờ biên: 1 cho pixel biên, 0 cho vùng phẳng
    edge_flag = np.zeros_like(image, dtype=np.uint8)
    edge_flag[edge_magnitude > edge_threshold] = 1
    
    # Tạo đệm cho ảnh
    padded = cv2.copyMakeBorder(image, large_window//2, large_window//2, large_window//2, large_window//2, cv2.BORDER_REFLECT)
    
    # Tạo mặt nạ để lưu vị trí nhiễu
    noise_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Quét từng pixel
    for i in range(large_window//2, padded.shape[0] - large_window//2):
        for j in range(large_window//2, padded.shape[1] - large_window//2):
            original_i = i - large_window//2
            original_j = j - large_window//2
            
            # Lấy vị trí trong ảnh gốc
            pixel_value = padded[i, j]
            
            # Chọn kích thước cửa sổ và ngưỡng phù hợp dựa trên ảnh cờ biên
            if edge_flag[original_i, original_j] == 1:  # Vùng biên
                window = padded[i - small_window//2:i + small_window//2 + 1, j - small_window//2:j + small_window//2 + 1]
                threshold = noise_threshold_edge
            else:  # Vùng phẳng
                window = padded[i - large_window//2:i + large_window//2 + 1, j - large_window//2:j + large_window//2 + 1]
                threshold = noise_threshold_flat
            
            median = np.median(window)
            
            # Phát hiện nhiễu dựa trên sự khác biệt với giá trị trung vị
            if abs(int(pixel_value) - median) > threshold:
                noise_mask[original_i, original_j] = 255  # Đánh dấu là nhiễu
    
    return noise_mask, edge_flag

# Hệ thống 2: Xác minh kết quả
def verify_noise_system2(image, noise_mask, edge_flag, window_size=5, similarity_threshold=10, ratio_threshold=0.3):
    # Tạo bản sao của mặt nạ nhiễu để chỉnh sửa
    verified_mask = np.copy(noise_mask)
    
    # Tạo đệm cho ảnh
    padded_image = cv2.copyMakeBorder(image, window_size//2, window_size//2, window_size//2, window_size//2, cv2.BORDER_REFLECT)
    padded_mask = cv2.copyMakeBorder(noise_mask, window_size//2, window_size//2, window_size//2, window_size//2, cv2.BORDER_CONSTANT, value=0)
    
    # Xác minh chỉ các pixel biên được đánh dấu là nhiễu
    for i in range(window_size//2, padded_image.shape[0] - window_size//2):
        for j in range(window_size//2, padded_image.shape[1] - window_size//2):
            original_i = i - window_size//2
            original_j = j - window_size//2
            
            # Chỉ xác minh các pixel biên được đánh dấu là nhiễu
            if noise_mask[original_i, original_j] == 255 and edge_flag[original_i, original_j] == 1:
                pixel_value = padded_image[i, j]
                
                # Lấy cửa sổ lân cận
                window = padded_image[i - window_size//2:i + window_size//2 + 1, j - window_size//2:j + window_size//2 + 1]
                mask_window = padded_mask[i - window_size//2:i + window_size//2 + 1, j - window_size//2:j + window_size//2 + 1]
                
                # Đếm số pixel nhiễu có giá trị gần với pixel đang xét
                similar_noise_pixels = 0
                total_noise_pixels = 0
                
                for m in range(window_size):
                    for n in range(window_size):
                        if mask_window[m, n] == 255:  # Nếu là pixel nhiễu theo Hệ thống 1
                            total_noise_pixels += 1
                            if abs(int(window[m, n]) - int(pixel_value)) <= similarity_threshold:
                                similar_noise_pixels += 1
                
                # Tính tỷ lệ R
                if total_noise_pixels > 0:
                    R = similar_noise_pixels / total_noise_pixels
                    
                    # Nếu tỷ lệ R nhỏ hơn ngưỡng, gỡ nhãn nhiễu
                    if R < ratio_threshold:
                        verified_mask[original_i, original_j] = 0  # Không còn là nhiễu
    
    return verified_mask

# Hàm lọc median chỉ áp dụng lên các pixel được đánh dấu là nhiễu
def apply_adaptive_median_filter(image, noise_mask, edge_flag, small_window=3, large_window=5):
    padded = cv2.copyMakeBorder(image, large_window//2, large_window//2, large_window//2, large_window//2, cv2.BORDER_REFLECT)
    result = np.copy(image)
    
    for i in range(large_window//2, padded.shape[0] - large_window//2):
        for j in range(large_window//2, padded.shape[1] - large_window//2):
            original_i = i - large_window//2
            original_j = j - large_window//2
            
            if noise_mask[original_i, original_j] == 255:  # Chỉ lọc những pixel bị phát hiện nhiễu
                # Chọn kích thước cửa sổ phù hợp dựa trên ảnh cờ biên
                if edge_flag[original_i, original_j] == 1:  # Vùng biên
                    window = padded[i - small_window//2:i + small_window//2 + 1, j - small_window//2:j + small_window//2 + 1]
                else:  # Vùng phẳng
                    window = padded[i - large_window//2:i + large_window//2 + 1, j - large_window//2:j + large_window//2 + 1]
                
                result[original_i, original_j] = np.median(window)
    
    return result


# img = cv2.imread(cv2.samples.findFile("ex5.jpg"), cv2.IMREAD_GRAYSCALE)
noisy_img= cv2.imread(cv2.samples.findFile("ex4.jpg"), cv2.IMREAD_GRAYSCALE)
# Sử dụng ảnh mẫu nếu không có ảnh đầu vào
# img = np.zeros((200, 200), dtype=np.uint8)
# Tạo gradient
# for i in range(200):
#     for j in range(200):
#         img[i, j] = min(255, (i + j) // 2)
# Vẽ một đường chéo
# cv2.line(img, (50, 50), (150, 150), 0, 2)

# Thêm nhiễu xung với xác suất 10%
# noisy_img = add_salt_pepper_noise(img, prob=0.1)

# Hệ thống 1: Phát hiện nhiễu dựa trên ảnh cờ biên
noise_mask_system1, edge_flag = detect_impulse_noise_system1(
    noisy_img, 
    edge_threshold=30, 
    noise_threshold_edge=50, 
    noise_threshold_flat=30, 
    small_window=3, 
    large_window=5
)

# Hệ thống 2: Xác minh kết quả
verified_mask = verify_noise_system2(
    noisy_img, 
    noise_mask_system1, 
    edge_flag, 
    window_size=5, 
    similarity_threshold=10, 
    ratio_threshold=0.3
)

# Lọc nhiễu bằng bộ lọc trung vị thích ứng
denoised_img = apply_adaptive_median_filter(
    noisy_img, 
    verified_mask, 
    edge_flag, 
    small_window=3, 
    large_window=5
)

# Hiển thị kết quả
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edge_flag * 255, cmap='gray')
plt.title("Ảnh cờ biên")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(noise_mask_system1, cmap='gray')
plt.title("Phát hiện nhiễu (Hệ thống 1)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(verified_mask, cmap='gray')
plt.title("Xác minh nhiễu (Hệ thống 2)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(denoised_img, cmap='gray')
plt.title("Ảnh đã lọc nhiễu")
plt.axis('off')

plt.tight_layout()
plt.show()