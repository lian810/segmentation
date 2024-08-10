from skimage.io import imread_collection
import matplotlib.pyplot as plt
from PIL import Image

seq = imread_collection("E:/threshold_segmentation/1.tif")
fig, ax = plt.subplots(figsize=[5, 5])
ax.imshow(seq[0])
plt.show()
fig, ax = plt.subplots(figsize=[5, 5])
ax.hist(seq[0].flatten(), bins=25, edgecolor='k')
plt.show()

im = seq[0]
im[(im < 50)] = 0
im[(im >= 50)] = 1

cmap = plt.get_cmap('plasma')
norm = plt.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=[5, 5])
ax.imshow(im, cmap=cmap, norm=norm)
ax.axis('off') 
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('temp.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.close(fig)
image = Image.open('temp.png')
image.save('temp.tiff', dpi=(600, 600))
