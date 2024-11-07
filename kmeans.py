import numpy as np
import cv2
from typing import Tuple

############################################################
#
#                       KMEANS
#
############################################################

def initialize_clusters(img: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Initialize cluster centers by randomly selecting pixels from the image.
    
    :param img (np.ndarray): The image array.
    :param num_clusters (int): The number of clusters to initialize.
    :return np.ndarray: Array of initial cluster centers.
    """
    # Randomly select pixels as initial cluster centers
    h, w, c = img.shape
    pixels = img.reshape(-1, c)
    random_indices = np.random.choice(pixels.shape[0], num_clusters, replace=False)
    initial_centers = pixels[random_indices].astype(np.float32)
    
    return initial_centers

def assign_clusters(img: np.ndarray, cluster_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Assign each pixel in the image to the nearest cluster center and calculate the overall distance.

    :param img (np.ndarray): The image array.
    :param cluster_centers (np.ndarray): Current cluster centers.   
    :return Tuple[np.ndarray, np.ndarray, float]: Tuple of the updated image, cluster mask, and overall distance.
    """
    h, w, c = img.shape
    pixels = img.reshape(-1, c)
    
    # Compute the distance from each pixel to each cluster center
    distances = np.linalg.norm(pixels[:, np.newaxis] - cluster_centers, axis=2)
    closest_clusters = np.argmin(distances, axis=1)
    
    # Update each pixel color to the nearest cluster color
    updated_pixels = cluster_centers[closest_clusters].astype(np.uint8)
    updated_img = updated_pixels.reshape(h, w, c)
    
    # Calculate overall distance (quantization error)
    overall_distance = np.sum(np.min(distances, axis=1))
    
    return updated_img, closest_clusters, overall_distance

def update_cluster_centers(img: np.ndarray, cluster_assignments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Update cluster centers as the mean of assigned pixels.

    :param img (np.ndarray): The image array.
    :param cluster_assignments (np.ndarray): Cluster assignments for each pixel.
    :param num_clusters (int): Number of clusters.
    :return np.ndarray: Updated cluster centers.
    """
    h, w, c = img.shape
    pixels = img.reshape(-1, c)
    
    # Compute the new cluster centers as the mean of pixels in each cluster
    new_centers = np.array([
        pixels[cluster_assignments == k].mean(axis=0) if np.any(cluster_assignments == k) else np.random.choice(pixels.shape[0], 1)
        for k in range(num_clusters)
    ])
    
    return new_centers

def kmeans_clustering(img: np.ndarray, num_clusters: int = 3, max_iterations: int = 100, tolerance: float = 0.01) -> np.ndarray:
    """
    Apply K-means clustering to do color quantization. Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less than the tolerance.
    
    :param img (np.ndarray): The image to be segmented.
    :param num_clusters (int): The number of clusters.
    :param max_iterations (int): The maximum number of iterations.
    :param tolerance (float): The convergence tolerance.
    :return np.ndarray: The segmented image.
    """
    # Step 1: Initialize clusters
    cluster_centers = initialize_clusters(img, num_clusters)
    previous_distance = float('inf')
    
    for i in range(max_iterations):
        # Step 2: Assign clusters
        updated_img, cluster_assignments, overall_distance = assign_clusters(img, cluster_centers)
        
        # Print the overall distance for each iteration
        print(f"Iteration {i+1}, Total Error: {overall_distance}")
        
        # Step 3: Update cluster centers
        cluster_centers = update_cluster_centers(img, cluster_assignments, num_clusters)
        
        # Check convergence (if the distance change is below tolerance)
        if abs(previous_distance - overall_distance) / previous_distance < tolerance:
            print(f"Convergence reached at iteration {i+1}")
            break
        previous_distance = overall_distance
    
    return updated_img

def load_and_process_image(file_path: str, scaling_factor: float = 0.5) -> np.ndarray:
    """
    Load and preprocess an image.
    
    :param file_path (str): Path to the image file.
    :param scaling_factor (float): Scaling factor to resize the image.        
    :return np.ndarray: The preprocessed image.
    """
    image = cv2.imread(file_path)

    # Scale the image to a smaller size for faster computation
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

def main():
    file_path = './graffiti.png'
    num_clusters = 16  # Experiment with different values of k
    
    img = load_and_process_image(file_path)
    segmented_img = kmeans_clustering(img, num_clusters)
    
    # Display original and segmented images
    cv2.imshow("Original", img)
    cv2.imshow("Color-based Segmentation K-means Clustering", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
