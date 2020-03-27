import numpy as np
from scipy.ndimage import label, find_objects


def segment(dataset):
    dataset = dataset.copy()
    background = dataset < 0
    particles = dataset >= 0
    dataset[background] = 0
    dataset[particles] = 1
    return dataset

def normalise(img):
    img = img.copy()
    img = img - img.min()
    img = img / img.max()
    return img

def split_image_into_128(img):
    arr = np.array(np.split(img, 16))
    arr = np.array(np.split(arr, 16, -1))
    arr = arr.reshape((-1, 128,128))[..., None]
    return arr
def segment_reshape_tiled_img(fit, name='myimg'):
    img = segment(fit)
    img = img.reshape((16, 16, 128, 128))
    img = np.concatenate(img, -1)
    img = np.concatenate(img, -2)
    return img
def save_png_true_fit(name, img, fit):
    img = img / img.max()
    imageio.imwrite(name + "_true.png", (255*img).astype('uint8'))
    imageio.imwrite(name + "_fit.png", (255*fit).astype('uint8'))
    

def calculate_center_of_mass(arr):
    """Find the center of mass of an array

    Parameters
    ----------
    arr : Numpy 2D Array

    Returns
    -------
    cx, cy: tuple of floats

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(10, 10))
    >>> data = afr.calculate_center_of_mass(arr)

    Notes
    -----
    This is a much simpler center of mass approach that the one from scipy.
    Gotten from stackoverflow:
    https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image

    """
    # Can consider subtracting minimum value
    # this gives the center of mass higher "contrast"
    # arr -= arr.min()
    arr = arr / np.sum(arr)

    dy = np.sum(arr, 1)
    dx = np.sum(arr, 0)

    (Y, X) = arr.shape
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    return cy, cx


def get_corners(slicepair, img):
    sy = slicepair[0]
    sx = slicepair[1]
    
    x0, xf = sx.start, sx.stop
    y0, yf = sy.start, sy.stop
    
    vec = np.array([[x0, y0], [xf, yf]])

    diff = (vec[1] - vec[0]) / 2
    coords = vec[0] + diff
    y, x = calculate_center_of_mass(img2[slicepair])
    coords2 = vec[0] + [x, y]
    return coords2    

# coords = np.array([get_corners(s, img) for s in slices])
# plt.figure(figsize = (40, 40))
# plt.imshow(img)
# plt.scatter(*coords.T, color='red')
# plt.savefig('img_with_scatter.png', dpi=200)


# import imageio
# img = imageio.imread("20200127 1321 STEM HAADF-DF4-DF2-BF 31.3 kx.tif")
# plot(*[img])

# arr = split_image_into_128(img)
# fit = model.predict(normalise(arr))
# retiled = segment_reshape_tiled_img(fit, 'nanoparticles')
# #save_png_true_fit("nano", img, retiled)
# plot(*[retiled])