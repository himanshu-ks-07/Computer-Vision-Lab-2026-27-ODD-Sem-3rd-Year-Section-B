import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os

# ---------------------------------------------------------
# Step 1 & 2: Load Image & Preprocessing
# ---------------------------------------------------------
os.makedirs('experiment6', exist_ok=True)
path = 'experiment6/sample.jpg'

if os.path.exists(path):
    print("Loading user image...")
    img = cv2.imread(path)
else:
    print("Sample image not found in 'experiment6/'. Creating sample synthetic image...")
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 50
    # Overlapping circles to test Watershed
    cv2.circle(img, (120, 150), 60, (200, 200, 200), -1)
    cv2.circle(img, (180, 150), 60, (250, 250, 250), -1)
    # Adding uneven gradient background
    gradient = np.tile(np.linspace(0, 100, 300, dtype=np.uint8), (300, 1))
    img[:, :, 0] = cv2.add(img[:, :, 0], gradient)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------------
# Step 3: Global Thresholding
# ---------------------------------------------------------
_, thresh_global = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# ---------------------------------------------------------
# Step 4: Otsu's Thresholding
# ---------------------------------------------------------
_, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------------------------------------------------------
# Step 5: Adaptive Thresholding
# ---------------------------------------------------------
thresh_adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

# ---------------------------------------------------------
# Step 6: Watershed Segmentation
# ---------------------------------------------------------
_, thresh_water = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh_water, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

watershed_img = img.copy()
markers = cv2.watershed(watershed_img, markers)
watershed_img[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# ---------------------------------------------------------
# Step 7: K-Means Clustering Segmentation
# ---------------------------------------------------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)

segmented_data = centers[labels.flatten()]
kmeans_img = segmented_data.reshape(img_rgb.shape)

# ---------------------------------------------------------
# Step 8 & 9: Save Visualizations
# ---------------------------------------------------------
titles = [
    'Original Image', 'Global Thresholding', "Otsu's Thresholding",
    'Adaptive Thresholding', 'Watershed Segmentation', 'K-Means Clustering (K=3)'
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), thresh_global, thresh_otsu,
    thresh_adaptive, cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB), kmeans_img
]

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i], fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.savefig('experiment6/segmentation_comparison.png')
plt.close()

print("Segmentation complete! Output saved as 'experiment6/segmentation_comparison.png'")