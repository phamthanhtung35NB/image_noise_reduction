import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm thêm nhiễu xung (salt & pepper noise) vào ảnh
def add_salt_pepper_noise(image, prob):
    noisy = np.copy(image)
    # Tạo ma trận ngẫu nhiên cùng kích thước với ảnh gốc
    rnd = np.random.rand(*image.shape)

    # Gán pixel trắng (255) với xác suất prob/2
    noisy[rnd < (prob / 2)] = 0  # "pepper" - nhiễu đen

    # Gán pixel đen (0) với xác suất prob/2
    noisy[rnd > 1 - (prob / 2)] = 255  # "salt" - nhiễu trắng

    return noisy

# Hàm phát hiện nhiễu bằng so sánh giá trị pixel với median của lân cận
def impulse_noise_detector(image, window_size=3, threshold=40):
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.uint8)

    # Quét từng pixel (trừ vùng viền)
    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            window = padded[i - 1:i + 2, j - 1:j + 2]  # Lấy cửa sổ 3x3
            median = np.median(window)  # Tính giá trị trung vị

            # Nếu giá trị pixel khác biệt quá lớn so với median → nghi là nhiễu
            if abs(int(padded[i, j]) - median) > threshold:
                output[i - 1, j - 1] = 255  # Đánh dấu là nhiễu (trắng)
            else:
                output[i - 1, j - 1] = 0  # Pixel bình thường

    return output

# Hàm lọc median chỉ áp dụng lên các pixel được đánh dấu là nhiễu
def apply_median_filter(image, mask, window_size=3):
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.copy(image)

    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            if mask[i - 1, j - 1] == 255:  # Chỉ lọc những pixel bị phát hiện nhiễu
                window = padded[i - 1:i + 2, j - 1:j + 2]
                result[i - 1, j - 1] = np.median(window)

    return result

# Đọc ảnh xám mẫu
img = cv2.imread(cv2.samples.findFile("ex5.jpg"), cv2.IMREAD_GRAYSCALE)

# Thêm nhiễu xung
noisy_img = add_salt_pepper_noise(img, prob=0.1)

# Phát hiện nhiễu
noise_mask = impulse_noise_detector(img)

# Khử nhiễu
denoised_img = apply_median_filter(noisy_img, noise_mask)

# Hiển thị ảnh
plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1); plt.imshow(img, cmap='gray'); plt.title("Ảnh gốc"); plt.axis('off')
# plt.subplot(1, 4, 2); plt.imshow(noisy_img, cmap='gray'); plt.title("Ảnh nhiễu"); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(noise_mask, cmap='gray'); plt.title("Mặt nạ nhiễu (phát hiện nhiễu)"); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(denoised_img, cmap='gray'); plt.title("Ảnh đã lọc nhiễu"); plt.axis('off')
plt.tight_layout()
plt.show()
