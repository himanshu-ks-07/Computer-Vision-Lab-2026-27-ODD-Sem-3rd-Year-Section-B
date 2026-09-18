import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure
import os
# ---------------------------------------------------------
# Step 1 & 2: Load Image and Preprocessing
# ---------------------------------------------------------
# Check uploaded images or auto-fallback to synthetic ones
path1 = 'experiment5/test1.jpg'
path2 = 'experiment5/test2.jpg'

if os.path.exists(path1) and os.path.exists(path2):
    print("Loading uploaded test images...")
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
else:
    print("Uploaded images not found in 'experiment5/'. Using generated sample images...")
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img1, (200, 200), 50, (128, 128, 128), -1)
    
    M = cv2.getRotationMatrix2D((150, 150), 15, 1.0)
    img2 = cv2.warpAffine(img1, M, (300, 300))

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Preprocessing
gray1_blur = cv2.GaussianBlur(gray1, (5, 5), 0)

# ---------------------------------------------------------
# Step 3 & 4: SIFT Feature Extraction & Keypoints
# ---------------------------------------------------------
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1_blur, None)

sift_image = cv2.drawKeypoints(
    img1, keypoints1, None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Save SIFT Output Image
cv2.imwrite('experiment5/sift_output.png', sift_image)
print("SIFT output saved as 'experiment5/sift_output.png'")

# ---------------------------------------------------------
# Step 5 & 6: HOG Feature Extraction
# ---------------------------------------------------------
hog_features, hog_image = hog(
    gray1_blur, 
    orientations=9, 
    pixels_per_cell=(8, 8), 
    cells_per_block=(2, 2), 
    visualize=True, 
    channel_axis=None
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Save HOG Output Image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT (Keypoints: {len(keypoints1)})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title(f'HOG (Features: {len(hog_features)})')
plt.axis('off')

plt.tight_layout()
plt.savefig('experiment5/sift_and_hog_comparison.png')
plt.close()
print("Comparison output saved as 'experiment5/sift_and_hog_comparison.png'")

# ---------------------------------------------------------
# Step 8: SIFT Image Matching
# ---------------------------------------------------------
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

if descriptors1 is not None and descriptors2 is not None:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m_match in matches:
        if len(m_match) == 2:
            m, n = m_match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    matched_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, good_matches[:30], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite('experiment5/matching_output.png', matched_img)
    print(f"Matching output saved as 'experiment5/matching_output.png' ({len(good_matches)} matches found)")

print("\n--- Processing Completed Successfully! ---")


#part-2

# Q1. What is feature extraction, and why is it important in computer vision?
# Answer:
# Feature extraction is a fundamental step in computer vision that transforms raw image data (pixel values) into a compact set of meaningful numerical vectors or descriptors.  Importance:Dimensionality Reduction: Reduces thousands/millions of raw pixels into concise representations while preserving key information.  Invariance: Extracted features stay stable despite changes in image scale, rotation, lighting, or viewpoint.  Performance Boost: Enables classification and recognition models to train significantly faster and avoid overfitting.

# Q2. Explain the working principle of Scale-Invariant Feature Transform (SIFT).  
# Answer:
# SIFT detects and describes local interest points through a 4-step pipeline:  Scale-Space Extrema Detection: Uses Difference-of-Gaussians (DoG) across image octaves to identify scale-invariant keypoints.  Keypoint Localization: Refines keypoint locations using Taylor series expansion and removes low-contrast or edge-response points.  Orientation Assignment: Assigns dominant gradient directions to keypoints to achieve rotation invariance.  Descriptor Generation: Computes gradient magnitudes/orientations in a $16 \times 16$ neighborhood around each keypoint to generate a 128-dimensional feature vector. 

# Q3. What are keypoints and feature descriptors in image analysis? 
# Answer:
# Keypoints (Interest Points): Specific spatial locations $(x, y)$ in an image that are unique and prominent, such as corners, blobs, or intersections.  Feature Descriptors: Numerical vectors calculated around keypoints that encode local texture, gradient, or edge patterns, allowing algorithms to match identical points across different images. 


# Q4. Explain the concept of Histogram of Oriented Gradients (HOG) and its significance.  
# Answer:
# Concept: HOG divides an image into small connected regions called cells (e.g., $8 \times 8$ pixels) and constructs a 1D histogram of gradient directions for pixels within each cell. These cells are then grouped into larger blocks for contrast normalization to handle illumination variations.  Significance: HOG effectively captures local shape and edge outlines while allowing small structural deformations, making it ideal for detecting structured objects like pedestrians and vehicles.  

# Q5. Compare SIFT and HOG based on robustness, computational complexity, and practical applications.  
# Answer:
# ParameterSIFT (Scale-Invariant Feature Transform)  PDFHOG (Histogram of Oriented Gradients)  PDFDetection TypeSparse keypoints/interest points  Dense, grid-based shape/contour extraction  Scale InvarianceHigh (Built-in Difference of Gaussian pyramid)  Low (Requires image pyramids manually)  Rotation InvarianceHigh (Assigns dominant orientation)  Low (Sensitive to image rotation)  Computational OverheadHigh (Iterative keypoint filtering)  Moderate (Fast matrix operations across grid cells)  ApplicationsPanorama stitching, 3D reconstruction, image matching  Pedestrian detection, vehicle detection, human pose estimation  

# Q6. Why is SIFT considered invariant to scale and rotation?  
# Answer:
# Scale Invariance: SIFT constructs a multi-scale Gaussian pyramid and calculates Difference-of-Gaussians (DoG). Keypoints are selected at the exact scale level where their response is highest, ensuring they are detected regardless of image zoom/scale.  Rotation Invariance: A dominant orientation is assigned to each keypoint based on local gradient directions. The descriptor frame is rotated relative to this orientation, making descriptor values independent of overall image rotation.  

# Q7. Mention three real-world applications where HOG descriptors are commonly used.  
# Answer:
# Pedestrian Detection: Used in ADAS (Autonomous Driving Systems) to identify pedestrians in traffic scenes.  Vehicle Tracking: Detecting cars and trucks in highway surveillance systems.  Gesture Recognition: Capturing hand shapes and static poses for human-computer interaction.  

# Q8. Why is feature extraction performed before image classification or object detection?  
# Answer:
# Reduces Redundant Data: Raw image pixels carry noise and uninformative background color info.  Improves Model Generalization: Feature descriptors summarize key structural shapes, making classifiers less sensitive to minor shifts, brightness, or scale differences.  Saves Memory and Time: Passing compact feature vectors instead of full high-res images speeds up training and inference time.  

# Q9. What are the advantages and limitations of handcrafted feature descriptors compared to deep learning-based feature extraction?  
# Answer:
# Handcrafted Features (SIFT, HOG):  Advantages: Fast to calculate, require no training datasets, highly interpretable.  Limitations: Require manual parameter tuning, underperform on complex non-rigid semantic patterns.  Deep Learning Features (CNNs):  Advantages: Automatically learn rich hierarchical representations directly from data; offer higher accuracy on complex datasets.  Limitations: Require massive labeled datasets, computationally expensive (need GPUs), act as "black box" models.  

# Q10. How do feature extraction techniques contribute to image matching, face recognition, and object detection systems?  
# Answer:
# Image Matching: Matching local descriptors (like SIFT) using nearest-neighbor algorithms (FLANN/BFMatcher) allows aligning and stitching images (panoramas) or 3D mapping.  Face Recognition: Extracting spatial geometry, edges, and textures creates unique face templates that can be indexed and verified against databases.  Object Detection: Techniques like HOG slide over image sub-windows to extract gradient vectors, which are then passed to classifiers (e.g., SVM) to locate objects with bounding boxes.  