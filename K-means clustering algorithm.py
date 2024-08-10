from skimage import io
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import Colorbar

image_path = 'pic.tif'
img = io.imread(image_path)

plt.imshow(img)
plt.show()
img_flat = img.reshape((-1, 3))

kmeans = KMeans(n_clusters=4, random_state=1)
labels = kmeans.fit_predict(img_flat)

counter = Counter(labels)

cluster0_count = counter[0]
cluster1_count = counter[1]
cluster2_count = counter[2]
cluster3_count = counter[3]

total_pixels = img_flat.shape[0]
cluster0_percentage = cluster0_count / total_pixels * 100
cluster1_percentage = cluster1_count / total_pixels * 100
cluster2_percentage = cluster2_count / total_pixels * 100
cluster3_percentage = cluster3_count / total_pixels * 100

print(f"Cluster 0 Percentage: {cluster0_percentage}%")
print(f"Cluster 1 Percentage: {cluster1_percentage}%")
print(f"Cluster 2 Percentage: {cluster2_percentage}%")
print(f"Cluster 3 Percentage: {cluster3_percentage}%")

cluster_centers = kmeans.cluster_centers_
segmented_img = labels.reshape(img.shape[:2])

for i, center in enumerate(cluster_centers):
    print(f"Cluster {i} Center (RGB): {center}")
    
cluster_ranges = []
for i in range(4):
    cluster_pixels = img_flat[labels == i]
    min_rgb = cluster_pixels.min(axis=0)
    max_rgb = cluster_pixels.max(axis=0)
    cluster_ranges.append((min_rgb, max_rgb))
    print(f"Cluster {i} RGB Range: Min {min_rgb}, Max {max_rgb}")

plt.figure(figsize=(8, 8))

plt.imshow(segmented_img, cmap='viridis')
img_plot = plt.imshow(segmented_img, cmap='viridis')
plt.title('Segmented Image')

cmap = img_plot.get_cmap()
norm = Normalize(vmin=segmented_img.min(), vmax=segmented_img.max())
colorbar = Colorbar(plt.gca(), img_plot, norm=norm, cmap=cmap, orientation='vertical')