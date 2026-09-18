#part-1

import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

# Step 1: Fetch grayscale image from online source
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

# Step 2 & 3: Compute DFT and Shift Zero-Frequency to center
dft = np.fft.fft2(img_gray)
dft_shift = np.fft.fftshift(dft)

# Step 4: Compute Magnitude Spectrum for visual analysis
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

# Mask setup dimensions
rows, cols = img_gray.shape
crow, ccol = rows // 2, cols // 2
radius = 30

# Step 5: Low-Pass Filter (Ideal LPF Mask)
lpf_mask = np.zeros((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
mask_area_lpf = (x - ccol) ** 2 + (y - crow) ** 2 <= radius**2
lpf_mask[mask_area_lpf] = 1

fshift_lpf = dft_shift * lpf_mask
img_lpf = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_lpf)))

# Step 6: High-Pass Filter (Ideal HPF Mask)
hpf_mask = np.ones((rows, cols), np.uint8)
hpf_mask[mask_area_lpf] = 0

fshift_hpf = dft_shift * hpf_mask
img_hpf = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_hpf)))

# Step 8 & 9: Plotting Results
titles = [
    "Original Grayscale",
    "Magnitude Spectrum",
    "Low-Pass Mask",
    "Low-Pass Filtered",
    "High-Pass Mask",
    "High-Pass Filtered",
]

images = [
    img_gray,
    magnitude_spectrum,
    lpf_mask,
    img_lpf,
    hpf_mask,
    img_hpf,
]

plt.figure(figsize=(14, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("exp4_output.png")
print(
    "Experiment 4 execution successful! Output saved as 'exp4_output.png'."
)

#part-2

# 1. What is the Fourier Transform? Why is it important in digital image processing?
# sol:
# The Fourier Transform is a mathematical technique that decomposes an image into its sine and cosine frequency components. It is important because it converts spatial intensity data into frequency information, allowing direct analysis and selective manipulation of slow transitions (low frequencies) and sharp variations/edges (high frequencies).

# 2. Differentiate between the spatial domain and the frequency domain.
# sol:
# Spatial Domain: Manipulates raw pixel intensity values directly using continuous grid coordinates $(x, y)$ and local spatial masks.
# Frequency Domain: Manipulates frequency components $(u, v)$ representing rates of intensity change across the image using Fourier spectra.

# 3. What is the significance of the Discrete Fourier Transform (DFT) in image processing?
# sol:
# DFT enables the practical digital computation of frequency spectra from discrete, sampled image pixels. It converts discrete spatial images into matrix-based frequency components, making computerized frequency filtering and spectral analysis possible.

# 4. Explain the purpose of shifting the zero-frequency component to the center of the frequency spectrum.
# sol:
# By default, the Discrete Fourier Transform places the DC component (zero-frequency, representing average image brightness) at the top-left corner $(0,0)$. Shifting it (np.fft.fftshift) moves $(0,0)$ to the matrix center, providing a symmetric, visually interpretable spectrum where low frequencies occupy the center and high frequencies spread outward.

# 5. Compare Low-Pass Frequency Filters and High-Pass Frequency Filters with suitable applications.
# sol:
# Low-Pass Frequency Filter (LPFF): Passes low frequencies (center of shifted spectrum) while blocking high frequencies. It is used for image smoothing, background blurring, and random noise suppression.High-Pass Frequency Filter (HPFF): Blocks low frequencies while passing high frequencies (outer regions of spectrum). It is used for edge detection, feature sharpening, and fine-detail extraction.

# 6. What is the role of the Inverse Fourier Transform (IDFT) in image reconstruction?
# sol:
# After frequency masks are applied to modify low or high frequencies, the Inverse Fourier Transform converts the processed frequency spectrum back into the spatial domain so the final filtered image can be rendered and viewed as regular pixels.

# 7. Why is frequency domain filtering preferred for certain image enhancement tasks?
# sol:
# Frequency domain filtering allows global pattern manipulation that is difficult in the spatial domain. It excels at removing periodic/patterned noise, precisely isolating specific frequency bands, and performing large-kernel operations efficiently via matrix multiplication instead of spatial convolution.

# 8. Mention four real-world applications where Fourier Transform is used in computer vision and image analysis.
# sol:
# Periodic Noise Reduction: Removing structured grid/stripe noise from scanned photos or satellite images.Biometric Systems: Extracting distinct frequency feature patterns from fingerprints and iris scans.Medical Diagnostic Imaging: Artifact suppression and structure enhancement in MRI and CT scans.Image Compression: Foundation algorithms for JPEG encoding and spectral image representation.

# 9. Compare frequency domain filtering with spatial domain filtering based on computational efficiency and practical applications.
# Computational Efficiency: Spatial domain is faster for small local operations ($3\times3$ or $5\times5$ matrices), while frequency domain is computationally efficient for large-scale or global filtering using FFT.Practical Applications: Spatial domain is ideal for quick, local real-time tasks (e.g., local edge detection), whereas frequency domain is best for global frequency isolation, periodic noise removal, and custom band-pass filtering.

# 10. How does frequency domain filtering improve the performance of image restoration and feature extraction techniques?
# sol:
# Frequency filtering cleanly decouples structural details from background noise based on rate-of-change signatures rather than pixel positions. This enables targeted removal of specific noise frequencies without distorting the rest of the image, leading to clearer edge extraction and restored images.
