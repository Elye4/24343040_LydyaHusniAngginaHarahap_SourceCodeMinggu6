import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import time
import os

# =========================
# 1. Load Citra
# =========================
img = cv2.imread('citra_asli.jpg', 0)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32)

os.makedirs("hasil", exist_ok=True)

# =========================
# 2. PSF Motion Blur
# =========================
def motion_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    psf[center, :] = 1

    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    psf = cv2.warpAffine(psf, M, (length, length))
    psf = psf / np.sum(psf)
    return psf

psf = motion_psf(15, 30)

# =========================
# 3. Degradasi
# =========================

# Motion Blur
blur = convolve2d(img, psf, 'same')

# Gaussian Noise + Blur
gauss_noise = np.random.normal(0, 20, img.shape)
gauss_blur = blur + gauss_noise

# Salt & Pepper + Blur
sp_blur = blur.copy()
prob = 0.05
rand = np.random.rand(*img.shape)
sp_blur[rand < prob/2] = 0
sp_blur[rand > 1 - prob/2] = 255

# =========================
# 4. Metode Restorasi
# =========================

def inverse_filter(img, psf):
    eps = 1e-3
    IMG = np.fft.fft2(img)
    PSF = np.fft.fft2(psf, s=img.shape)
    result = np.fft.ifft2(IMG / (PSF + eps))
    return np.abs(result)

def wiener_filter(img, psf, K=0.01):
    IMG = np.fft.fft2(img)
    PSF = np.fft.fft2(psf, s=img.shape)
    PSF_conj = np.conj(PSF)
    result = np.fft.ifft2((PSF_conj / (np.abs(PSF)**2 + K)) * IMG)
    return np.abs(result)

def lucy_richardson(img, psf, iterations=10):
    estimate = np.full(img.shape, 0.5)
    psf_mirror = psf[::-1, ::-1]

    for i in range(iterations):
        conv = convolve2d(estimate, psf, 'same')
        relative_blur = img / (conv + 1e-5)
        estimate *= convolve2d(relative_blur, psf_mirror, 'same')

    return estimate

# =========================
# 5. Evaluasi
# =========================
def evaluate(original, restored):
    # pastikan nilai 0–255
    original = np.clip(original, 0, 255)
    restored = np.clip(restored, 0, 255)

    # ubah ke float64 biar aman
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    return (
        mse(original, restored),
        psnr(original, restored, data_range=255),
        ssim(original, restored, data_range=255)
    )

# =========================
# 6. Restorasi + Waktu
# =========================
start = time.time()

# Motion blur
inv_blur = inverse_filter(blur, psf)
wie_blur = wiener_filter(blur, psf)
lucy_blur = lucy_richardson(blur, psf)

# Gaussian + blur
inv_g = inverse_filter(gauss_blur, psf)
wie_g = wiener_filter(gauss_blur, psf)
lucy_g = lucy_richardson(gauss_blur, psf)

# Salt pepper + blur
inv_sp = inverse_filter(sp_blur, psf)
wie_sp = wiener_filter(sp_blur, psf)
lucy_sp = lucy_richardson(sp_blur, psf)

end = time.time()

print("Waktu komputasi:", end - start)

# =========================
# 7. Evaluasi Output
# =========================
print("\n=== MOTION BLUR ===")
print("Inverse:", evaluate(img, inv_blur))
print("Wiener:", evaluate(img, wie_blur))
print("Lucy:", evaluate(img, lucy_blur))

print("\n=== GAUSSIAN + BLUR ===")
print("Inverse:", evaluate(img, inv_g))
print("Wiener:", evaluate(img, wie_g))
print("Lucy:", evaluate(img, lucy_g))

print("\n=== SALT PEPPER + BLUR ===")
print("Inverse:", evaluate(img, inv_sp))
print("Wiener:", evaluate(img, wie_sp))
print("Lucy:", evaluate(img, lucy_sp))

# =========================
# 8. Simpan Gambar
# =========================
def save_image(name, image):
    image = np.clip(image, 0, 255)
    cv2.imwrite(f"hasil/{name}.jpg", image.astype(np.uint8))

save_image("asli", img)
save_image("motion_blur", blur)
save_image("gaussian_blur", gauss_blur)
save_image("sp_blur", sp_blur)

save_image("inv_gaussian", inv_g)
save_image("wiener_gaussian", wie_g)
save_image("lucy_gaussian", lucy_g)

# =========================
# 9. Tampilkan Gambar 
# =========================
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Citra Asli")

plt.subplot(2,3,2)
plt.imshow(blur, cmap='gray')
plt.title("Motion Blur")

plt.subplot(2,3,3)
plt.imshow(gauss_blur, cmap='gray')
plt.title("Gaussian + Blur")

plt.subplot(2,3,4)
plt.imshow(inv_g, cmap='gray')
plt.title("Inverse")

plt.subplot(2,3,5)
plt.imshow(wie_g, cmap='gray')
plt.title("Wiener")

plt.subplot(2,3,6)
plt.imshow(lucy_g, cmap='gray')
plt.title("Lucy")

plt.tight_layout()
plt.show()