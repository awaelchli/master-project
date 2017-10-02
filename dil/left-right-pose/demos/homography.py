import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider
from dataimport.ImageNet import apply_homography, homography_roty


def update(val):
    hom = homography_roty(val, w, h, 0.7)
    rotated = apply_homography(image, hom)
    ax1.imshow(rotated, cmap=plt.cm.gray)

image = Image.open('fish.JPEG')
w, h = image.width, image.height

fig, (ax1, ax3) = plt.subplots(1, 2)

ax1.imshow(image, cmap=plt.cm.gray)

slider = Slider(ax3, 'Angle', -90, 90, valinit=0)
slider.on_changed(update)

plt.show()


