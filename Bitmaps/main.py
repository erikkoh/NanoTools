"""
Main code for generating the bitmap images used in the FIB
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from shapely.geometry import Polygon, Point
from PIL import Image
plt.rcParams['image.cmap'] = 'gray'


def get_compass(N, scale, offset= 60):
    """
    Simple function to create a magnetic compass to align the external magnetic field in the FIB
    """
    #The 7 here is mostly an artifact from the original code unsure how important it still is
    size = scale*(N+7)                   
    x = np.arange(-offset-size,size+1+offset)
    y = np.arange(-offset-size,size+1+offset)
    xx,yy = np.meshgrid(x,y,indexing='ij')
    return (xx**2+yy**2) > size**2


def create_polygon(num_sides, radius):
    """
    Create a binary mask of a regular polygon of arbitrary number of sides.
    """
    radius = radius + 7
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertices = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    poly = Polygon(vertices)

    # Create a grid to represent the polygon
    min_x, min_y, max_x, max_y = poly.bounds
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    
    # Create a binary mask
    grid = np.zeros((height, width), dtype=int)

    # Fill the grid with the polygon shape
    for x in range(width):
        for y in range(height):
            if poly.contains(Point(x + min_x, y + min_y)):
                grid[y, x] = 1

    return grid


#This was mostly stripped from the provided notbooks with small adjustments to fit the spesific problem
def get_grid(sh,offset,spacing, f, size=1, num_sides = 3, start = 0):
    """ 
    Convenient for creating a millingpattern for 1D investigations.
    Parameters:
    -------------
        sh, offset, spacing

        f:
            Function that creates the different elements. The grid will change the argument left to right.
        size:
            Passed to f determines the size of the elements.
        num_sides :
            Passed to f determines the number of sides of the first row elements ie. 3 => triangles.
    """
    ngrid= np.zeros(sh)
    reference = np.zeros(sh)
    for i,j in enumerate(np.arange(offset, sh[0]-offset, spacing)):
        for k,l in enumerate(np.arange(offset, sh[1]-offset, spacing)):
            element = f(k+num_sides,(i+1)*size+start)
            ex,ey = element.shape
            ngrid[l-ex//2:l+ex//2 + ex%2,j-ey//2:j+ey//2+ey%2] = element
            reference[l,j] = 1
    com = (np.array(nd.center_of_mass(reference)) - np.array(sh)//2).astype(int)
    return np.roll(np.invert(ngrid.astype(bool)), -com,axis=[0,1])


def Test_plotting(sh=(512,512), offset=50, spacing=50, f = create_polygon, size=1, num_sides=3):
    plt.figure()
    plt.imshow(get_grid(sh,offset,spacing, f, size, num_sides))
    plt.show()

def save_numpy_to_bmp(arr, path):
    """
    Save a numpy array as a BMP image using PIL.Image.
    Converts boolean or integer arrays to uint8 for saving.
    """
    arr = arr.astype('uint8') * 255 if arr.dtype == bool else arr.astype('uint8')
    img = Image.fromarray(arr)
    img = img.convert("RGB")
    img.save(path)

def save_bitmaps(sh=(512,512), offset=60, spacing=120, f = create_polygon):
    os.makedirs("./image_files", exist_ok=True)
    dst_folder = os.path.join(os.getcwd(), 'image_files')
    for i in range(2):
        mask = get_grid(sh,offset,spacing,f, size=5, num_sides=3+i*3)
        fname = f'0{i}_shapes_l_shape{mask.shape}_numer_of_sides{3+i*3}.bmp'
        save_numpy_to_bmp(mask, os.path.join(dst_folder, fname))
    for i in range(2):
        mask = get_grid(sh,offset,spacing,f, size=5, num_sides=3+i*3, start= 20)
        fname = f'1{i}_shapes_r_shape{mask.shape}_numer_of_sides{3+i*3}.bmp'
        save_numpy_to_bmp(mask, os.path.join(dst_folder, fname))
    compass = get_compass(50, 3, 60)
    save_numpy_to_bmp(compass, os.path.join(dst_folder, '00_compass_{compass.shape}.bmp'))

""" Test_plotting() """
save_bitmaps()