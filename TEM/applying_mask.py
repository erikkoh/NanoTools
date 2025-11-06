from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from shapely.geometry import Polygon, Point

im = Image.open('TEM/DPC_pictures/010.png').convert('RGB')
plt.imshow(np.asarray(im))
plt.show()
place_holder = Image.new('RGB', im.size, (0, 0, 0))
mask = Image.open('image_files/00_shapes_l_shape(512, 512)_numer_of_sides3.bmp')
mask = PIL.ImageOps.invert(mask.resize(im.size)).convert('L').transpose(Image.TRANSPOSE).rotate(180)
offset_x = 6
offset_y = 6
mask = mask.transform(mask.size, Image.AFFINE, (1, 0, offset_x, 0, 1, offset_y))

final_image = Image.composite(im,place_holder, mask=mask)
plt.imshow(np.asarray(final_image))
plt.show()