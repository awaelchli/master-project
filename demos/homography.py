import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider
from dataimport.ImageNet import rotate_image


def update(val):
    rotated = rotate_image(image, val, z=0.7)
    ax1.imshow(rotated, cmap=plt.cm.gray)

image = Image.open('fish.JPEG')

fig, (ax1, ax3) = plt.subplots(1, 2)

ax1.imshow(rotate_image(image, 0), cmap=plt.cm.gray)

slider = Slider(ax3, 'Angle', -90, 90, valinit=0)
slider.on_changed(update)

plt.show()


