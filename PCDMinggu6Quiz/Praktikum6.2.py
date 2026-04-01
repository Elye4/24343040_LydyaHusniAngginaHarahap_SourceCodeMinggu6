import numpy as np
import cv2
import matplotlib.pyplot as plt

def praktikum_6_2():
    print("\nPRAKTIKUM 6.2: INVERSE FILTER VS WIENER FILTER")
    print("=" * 50)
    
    def create_degraded_image():
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (100, 100), 200, -1)
        cv2.circle(img, (180, 80), 40, 150, -1)

        blurred = cv2.GaussianBlur(img, (9, 9), 2)
        noise = np.random.normal(0, 15, blurred.shape)
        degraded = np.clip(blurred + noise, 0, 255)

        return img, degraded.astype(np.uint8)

    def inverse_filter(degraded, psf, epsilon=1e-3):
        pad_size = psf.shape[0] // 2
        padded = cv2.copyMakeBorder(degraded, pad_size, pad_size, 
                                   pad_size, pad_size, cv2.BORDER_REFLECT)

        G = np.fft.fft2(padded.astype(float))

        psf_padded = np.zeros_like(padded, dtype=float)
        cy, cx = psf.shape[0]//2, psf.shape[1]//2
        py, px = padded.shape[0]//2, padded.shape[1]//2

        psf_padded[py-cy:py-cy+psf.shape[0], px-cx:px-cx+psf.shape[1]] = psf
        psf_padded = np.fft.ifftshift(psf_padded)

        H = np.fft.fft2(psf_padded)
        H[np.abs(H) < epsilon] = epsilon

        F_hat = G / H
        restored = np.abs(np.fft.ifft2(F_hat))

        return np.clip(restored[pad_size:-pad_size, pad_size:-pad_size], 0, 255).astype(np.uint8)

    def wiener_filter(degraded, psf, K=0.01):
        pad_size = psf.shape[0] // 2
        padded = cv2.copyMakeBorder(degraded, pad_size, pad_size, 
                                   pad_size, pad_size, cv2.BORDER_REFLECT)

        G = np.fft.fft2(padded.astype(float))

        psf_padded = np.zeros_like(padded, dtype=float)
        cy, cx = psf.shape[0]//2, psf.shape[1]//2
        py, px = padded.shape[0]//2, padded.shape[1]//2

        psf_padded[py-cy:py-cy+psf.shape[0], px-cx:px-cx+psf.shape[1]] = psf
        psf_padded = np.fft.ifftshift(psf_padded)

        H = np.fft.fft2(psf_padded)
        H_conj = np.conj(H)

        W = H_conj / (np.abs(H)**2 + K)

        F_hat = G * W
        restored = np.abs(np.fft.ifft2(F_hat))

        return np.clip(restored[pad_size:-pad_size, pad_size:-pad_size], 0, 255).astype(np.uint8)

    def create_gaussian_psf(size=9, sigma=2):
        ax = np.arange(-size//2 + 1., size//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return psf / np.sum(psf)

    original, degraded = create_degraded_image()
    psf = create_gaussian_psf(9, 2)

    restorations = {}

    for eps in [1e-2, 1e-3, 1e-4]:
        restorations[f'Inverse ε={eps}'] = inverse_filter(degraded, psf, eps)

    for K in [0.1, 0.01, 0.001]:
        restorations[f'Wiener K={K}'] = wiener_filter(degraded, psf, K)

    total_images = 3 + len(restorations)
    cols = 3
    rows = int(np.ceil(total_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.ravel()

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(degraded, cmap='gray')
    axes[1].set_title('Degraded')
    axes[1].axis('off')

    axes[2].imshow(psf, cmap='hot')
    axes[2].set_title('PSF')
    axes[2].axis('off')

    for i, (title, img) in enumerate(restorations.items()):
        axes[i+3].imshow(img, cmap='gray')
        axes[i+3].set_title(title)
        axes[i+3].axis('off')

    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nHASIL PSNR")
    print("-"*40)

    def safe_psnr(original, test):
        mse = np.mean((original.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return 100
        return 10 * np.log10(255**2 / mse)

    for title, img in restorations.items():
        print(f"{title}: {safe_psnr(original, img):.2f} dB")

    return original, degraded, restorations

original, degraded, restorations = praktikum_6_2()