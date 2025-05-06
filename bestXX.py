import cv2
import numpy as np
import matplotlib.pyplot as plt
# Tham số phương pháp gốc (có thể điều chỉnh)
SMALL_WIN = 3       # W_k (cửa sổ nhỏ)
LARGE_WIN = 7       # W_l (cửa sổ lớn)
TE = 20             # Ngưỡng phát hiện biên (edge threshold)
T_EDGE = 30         # Ngưỡng phát hiện nhiễu ở vùng biên
T_FLAT = 50         # Ngưỡng phát hiện nhiễu ở vùng phẳng
TC = 20             # Ngưỡng số pixel kiểm tra trong System 2
TN = 10             # Biên độ biến thiên nhiễu
TR = 0.8            # Tỷ lệ tương tự để xác minh
VER_WIN = 9         # Kích thước cửa sổ xác minh (System 2)


def detect_impulse_system1(img_gray):
    """
    Hệ thống 1: Phát hiện nhiễu dựa trên edge flag image
    Trả về:
      - edge_flag: ảnh cờ biên (0/1)
      - noise_flag: cờ nhiễu ban đầu (0/1)
    """
    # Tính median với cửa sổ nhỏ và lớn
    med_small = cv2.medianBlur(img_gray, SMALL_WIN)
    med_large = cv2.medianBlur(img_gray, LARGE_WIN)

    # Tạo ảnh cờ biên: pixel nào có sự khác biệt giữa hai median vượt TE
    edge_flag = (np.abs(med_small.astype(int) - med_large.astype(int)) > TE).astype(np.uint8)

    # Phát hiện nhiễu ban đầu
    diff_small = np.abs(img_gray.astype(int) - med_small.astype(int))
    diff_large = np.abs(img_gray.astype(int) - med_large.astype(int))

    # Pixel vùng biên: dùng ngưỡng T_EDGE, vùng phẳng dùng T_FLAT
    noise_flag = np.zeros_like(img_gray, dtype=np.uint8)
    # Vùng biên
    mask_edge = edge_flag == 1
    noise_flag[mask_edge & (diff_small > T_EDGE)] = 1
    # Vùng phẳng
    mask_flat = edge_flag == 0
    noise_flag[mask_flat & (diff_large > T_FLAT)] = 1

    return edge_flag, noise_flag


def verify_system2(img_gray, edge_flag, noise_flag):
    """
    Hệ thống 2: Xác minh các pixel nhiễu ở vùng biên
    """
    # Dilation edge_flag để xác định vùng biên mở rộng
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    expanded_edge = cv2.dilate(edge_flag, kernel, iterations=1)

    verified = noise_flag.copy()
    pad = VER_WIN // 2
    padded_img = cv2.copyMakeBorder(img_gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    padded_noise = cv2.copyMakeBorder(noise_flag, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            # Chỉ xét pixel đã đánh dấu nhiễu ban đầu và trong vùng biên mở rộng
            if noise_flag[i, j] == 1 and expanded_edge[i, j] == 1:
                # Vùng xác minh
                win_noise = padded_noise[i:i+VER_WIN, j:j+VER_WIN]
                win_img = padded_img[i:i+VER_WIN, j:j+VER_WIN].astype(int)

                # Chọn các pixel đã đánh dấu nhiễu nằm trong vùng phẳng lân cận
                # (ban đầu noise_flag==1 nhưng edge_flag==0)
                candidates = []
                for r in range(VER_WIN):
                    for c in range(VER_WIN):
                        orig_r = i + r - pad
                        orig_c = j + c - pad
                        if (win_noise[r, c] == 1) and (edge_flag[orig_r, orig_c] == 0):
                            candidates.append(win_img[r, c])
                            if len(candidates) >= TC:
                                break
                    if len(candidates) >= TC:
                        break

                # Nếu không thu đủ TC pixel, dùng tất cả pixel noise_flag trong cửa sổ
                if len(candidates) == 0:
                    # Không có pixel phẳng nào để so sánh => giữ nguyên đánh dấu
                    continue

                # Tính tỷ lệ pixel có cường độ khác không quá TN so với pixel trung tâm
                center = int(img_gray[i, j])
                similar = sum(abs(val - center) <= TN for val in candidates)
                R = similar / len(candidates)
                if R < TR:
                    # Nhận nhầm, bỏ cờ nhiễu
                    verified[i, j] = 0

    return verified


def denoise_using_mask(img_gray, noise_mask):
    """
    Lọc nhiễu: thay mỗi pixel nhiễu bằng giá trị median của các pixel tốt lân cận
    """
    result = img_gray.copy()
    rows, cols = img_gray.shape
    pad = LARGE_WIN // 2
    padded = cv2.copyMakeBorder(img_gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    padded_mask = cv2.copyMakeBorder(noise_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for i in range(rows):
        for j in range(cols):
            if noise_mask[i, j] == 1:
                # Lấy cửa sổ lớn bao quanh
                win_vals = []
                for r in range(LARGE_WIN):
                    for c in range(LARGE_WIN):
                        if padded_mask[i+r, j+c] == 0:
                            win_vals.append(int(padded[i+r, j+c]))
                if win_vals:
                    result[i, j] = np.median(win_vals)
                else:
                    # Không có pixel tốt xung quanh => dùng median chung
                    result[i, j] = cv2.medianBlur(img_gray, LARGE_WIN)[i, j]
    return result


def full_restoration(img_gray):
    # 1. System 1: phát hiện biên và nhiễu ban đầu
    edge_flag, noise_flag1 = detect_impulse_system1(img_gray)
    # 2. System 2: xác minh
    noise_flag2 = verify_system2(img_gray, edge_flag, noise_flag1)
    # 3. Khôi phục ảnh
    restored = denoise_using_mask(img_gray, noise_flag2)
    return {
        'edge_flag': edge_flag,
        'initial_noise': noise_flag1,
        'verified_noise': noise_flag2,
        'restored': restored
    }
def show_results(original, outputs):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(outputs['edge_flag'], cmap='gray')
    axs[1].set_title('Edge Flag')
    axs[2].imshow(outputs['initial_noise'], cmap='gray')
    axs[2].set_title('Initial Noise')
    axs[3].imshow(outputs['verified_noise'], cmap='gray')
    axs[3].set_title('Verified Noise')
    axs[4].imshow(outputs['restored'], cmap='gray')
    axs[4].set_title('Restored')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('ex2.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError('Không tìm thấy input.png')
    outputs = full_restoration(img)
    show_results(img, outputs)

# if __name__ == '__main__':
#     img = cv2.imread('ex3.png', cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError('Không tìm thấy input.png')
#     outputs = full_restoration(img)
#     cv2.imwrite('edge_flag.png', outputs['edge_flag'] * 255)
#     cv2.imwrite('noise_initial.png', outputs['initial_noise'] * 255)
#     cv2.imwrite('noise_verified.png', outputs['verified_noise'] * 255)
#     cv2.imwrite('restored.png', outputs['restored'])


# if __name__ == '__main__':
#     # Ví dụ sử dụng:
#     img = cv2.imread('ex2.jpg', cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError('Không tìm thấy lena.png')

#     outputs = full_restoration(img)

#     # Hiển thị các ảnh
#     cv2.imshow('Original Image', img)
#     cv2.imshow('Edge Flag', outputs['edge_flag'] * 255)
#     cv2.imshow('Initial Noise Flag', outputs['initial_noise'] * 255)
#     cv2.imshow('Verified Noise Flag', outputs['verified_noise'] * 255)
#     cv2.imshow('Restored Image', outputs['restored'])

#     # Chờ nhấn phím
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
