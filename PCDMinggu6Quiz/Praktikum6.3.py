import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def praktikum_6_3():
    print("\nPRAKTIKUM 6.3: MOTION BLUR ESTIMATION DAN DEBLURRING")
    print("=" * 60)
    
    def create_motion_blurred_image():
        img = np.zeros((256, 256), dtype=np.uint8)
        
        for i in range(0, 256, 20):
            cv2.line(img, (i, 0), (i, 255), 150, 1)
            cv2.line(img, (0, i), (255, i), 150, 1)
        
        cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
        cv2.circle(img, (180, 100), 30, 180, -1)
        
        cv2.putText(img, 'TEST', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 220, 2)
        
        length = 21
        angle = 30
        
        psf = np.zeros((length, length))
        center = length // 2
        angle_rad = np.deg2rad(angle)
        
        x_start = int(center - (length/2) * np.cos(angle_rad))
        y_start = int(center - (length/2) * np.sin(angle_rad))
        x_end = int(center + (length/2) * np.cos(angle_rad))
        y_end = int(center + (length/2) * np.sin(angle_rad))
        
        cv2.line(psf, (x_start, y_start), (x_end, y_end), 1, 1)
        psf = psf / np.sum(psf)
        
        blurred = cv2.filter2D(img.astype(float), -1, psf)
        noise = np.random.normal(0, 10, blurred.shape)
        blurred_noisy = np.clip(blurred + noise, 0, 255)
        
        return img, blurred_noisy.astype(np.uint8), psf
    
    def estimate_motion_blur_parameters(image):
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for line in lines[:5]:
                rho, theta = line[0]
                angles.append(np.degrees(theta))
        
        if angles:
            estimated_angle = np.mean(angles) % 180
        else:
            estimated_angle = 0
        
        autocorr = signal.correlate2d(image, image, mode='same')
        autocorr = autocorr / np.max(autocorr)
        
        center = autocorr.shape[0] // 2
        profile = autocorr[center, :]
        
        half_max = 0.5
        above_half = profile > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = indices[-1] - indices[0]
            estimated_length = int(max(1, fwhm // 2))
        else:
            estimated_length = 15
        
        return estimated_length, estimated_angle
    
    def motion_deblur_inverse(image, length, angle, epsilon=1e-3):
        psf_size = max(31, 2 * int(length) + 1)
        psf = np.zeros((psf_size, psf_size))
        center = psf_size // 2
        angle_rad = np.deg2rad(angle)
        
        x_start = int(center - (length/2) * np.cos(angle_rad))
        y_start = int(center - (length/2) * np.sin(angle_rad))
        x_end = int(center + (length/2) * np.cos(angle_rad))
        y_end = int(center + (length/2) * np.sin(angle_rad))
        
        cv2.line(psf, (x_start, y_start), (x_end, y_end), 1, 1)
        psf = psf / np.sum(psf)
        
        pad_size = psf_size // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        
        G = np.fft.fft2(padded.astype(float))
        
        psf_padded = np.zeros_like(padded, dtype=float)
        psf_center = psf_size // 2
        pad_center = padded.shape[0] // 2
        
        psf_padded[pad_center-psf_center:pad_center-psf_center+psf_size,
                   pad_center-psf_center:pad_center-psf_center+psf_size] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        
        H = np.fft.fft2(psf_padded)
        H[np.abs(H) < epsilon] = epsilon
        
        F_hat = G / H
        restored = np.abs(np.fft.ifft2(F_hat))
        restored = restored[pad_size:-pad_size, pad_size:-pad_size]
        
        return np.clip(restored, 0, 255).astype(np.uint8), psf
    
    def motion_deblur_wiener(image, length, angle, K=0.001):
        psf_size = max(31, 2 * int(length) + 1)
        psf = np.zeros((psf_size, psf_size))
        center = psf_size // 2
        angle_rad = np.deg2rad(angle)
        
        x_start = int(center - (length/2) * np.cos(angle_rad))
        y_start = int(center - (length/2) * np.sin(angle_rad))
        x_end = int(center + (length/2) * np.cos(angle_rad))
        y_end = int(center + (length/2) * np.sin(angle_rad))
        
        cv2.line(psf, (x_start, y_start), (x_end, y_end), 1, 1)
        psf = psf / np.sum(psf)
        
        pad_size = psf_size // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        
        G = np.fft.fft2(padded.astype(float))
        
        psf_padded = np.zeros_like(padded, dtype=float)
        psf_center = psf_size // 2
        pad_center = padded.shape[0] // 2
        
        psf_padded[pad_center-psf_center:pad_center-psf_center+psf_size,
                   pad_center-psf_center:pad_center-psf_center+psf_size] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        
        H = np.fft.fft2(psf_padded)
        H_conj = np.conj(H)
        
        W = H_conj / (np.abs(H)**2 + K)
        F_hat = G * W
        
        restored = np.abs(np.fft.ifft2(F_hat))
        restored = restored[pad_size:-pad_size, pad_size:-pad_size]
        
        return np.clip(restored, 0, 255).astype(np.uint8), psf
    
    def richardson_lucy_deblur(image, psf, iterations=20):
        image = image.astype(np.float32)
        psf = psf.astype(np.float32)
        
        estimate = image.copy()
        psf_flipped = np.flip(psf)
        
        for i in range(iterations):
            conv = cv2.filter2D(estimate, -1, psf)
            conv = np.where(conv == 0, 1e-8, conv)
            ratio = image / conv
            correction = cv2.filter2D(ratio, -1, psf_flipped)
            estimate = estimate * correction
            estimate = np.clip(estimate, 0, 255)
        
        return estimate.astype(np.uint8)
    
    original, blurred, true_psf = create_motion_blurred_image()
    
    estimated_length, estimated_angle = estimate_motion_blur_parameters(blurred)
    true_length = 21
    true_angle = 30
    
    print(f"True Blur Parameters: Length = {true_length}, Angle = {true_angle}°")
    print(f"Estimated Parameters: Length = {estimated_length}, Angle = {estimated_angle:.1f}°")
    
    deblurring_results = {}
    
    inv_deblurred, est_psf_inv = motion_deblur_inverse(blurred, estimated_length, estimated_angle)
    deblurring_results['Inverse Filter'] = inv_deblurred
    
    wiener_deblurred, est_psf_wiener = motion_deblur_wiener(blurred, estimated_length, estimated_angle)
    deblurring_results['Wiener Filter'] = wiener_deblurred
    
    wiener_true, _ = motion_deblur_wiener(blurred, true_length, true_angle)
    deblurring_results['Wiener (True Params)'] = wiener_true
    
    rl_deblurred = richardson_lucy_deblur(blurred, true_psf, iterations=20)
    deblurring_results['Richardson-Lucy'] = rl_deblurred
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(blurred, cmap='gray')
    mse_blurred = np.mean((original.astype(float) - blurred.astype(float)) ** 2)
    psnr_blurred = 10 * np.log10(255**2 / mse_blurred)
    axes[0,1].set_title(f'Motion Blurred\nPSNR: {psnr_blurred:.2f} dB')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(true_psf, cmap='hot')
    axes[0,2].set_title('True PSF')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(est_psf_inv, cmap='hot')
    axes[0,3].set_title('Estimated PSF')
    axes[0,3].axis('off')
    
    for idx, (method_name, result) in enumerate(deblurring_results.items()):
        mse = np.mean((original.astype(float) - result.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        axes[1, idx].imshow(result, cmap='gray')
        axes[1, idx].set_title(f'{method_name}\nPSNR: {psnr:.2f}')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original, blurred, deblurring_results

original_motion, blurred_motion, deblurring_results = praktikum_6_3()