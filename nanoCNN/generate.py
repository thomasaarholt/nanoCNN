
import itertools

import numpy as np
import matplotlib.pyplot as plt
from pyellipsoid.drawing import make_ellipsoid_image
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate as ndimagerotate
import imageio

from pathlib import Path
from tqdm.auto import tqdm


plt.rcParams['figure.max_open_warning'] = 2000

def rotation_matrix(angles = (0, 0, 0), degrees=True):
    return R.from_euler('zyx', angles, degrees=degrees).as_dcm()

def plot(*imgs):
    fig, AX = plt.subplots(ncols=len(imgs), squeeze=False, constrained_layout=True)
    for i, img in enumerate(imgs):
        AX[0, i].imshow(img)
    fig.set_figwidth(4*len(imgs))

def generate_random_ellipsoid(xysize=61, also_make_with_higher_radii=False, higher_radii=2):
    shape = np.array((xysize, xysize, xysize))
    # Define an ellipsoid, axis order is: X, Y, Z
    center = shape / 2
    radii = xysize/2*(np.random.random(size=3)+1)/2 #/ 2**0.5
    if also_make_with_higher_radii:
        radii -= higher_radii

    angles = np.deg2rad(np.random.random(size=3)*180) # Order of rotations is X, Y, Z
    
    normal_ellipsoid = make_ellipsoid_image(shape, center, radii, angles).sum(-1)
    if also_make_with_higher_radii:
        larger_ellipsoid = make_ellipsoid_image(shape, center, radii+higher_radii, angles).sum(-1)
    else:
        larger_ellipsoid = None
    
    return normal_ellipsoid, larger_ellipsoid

def get_radius(A, B):
    if len(B) == 0:
        return np.array([999])
    A, B = np.array(A), np.array(B)
    return np.sqrt(((A - B)**2).sum(-1))


def ellipsoid_and_tetrahedron_image(N_particles=10, shape=(128, 128), higher_radii=2, minsize=5, maxsize=40, fraction_tetra=1/3, fraction_cube=1/3, no_overlap=False, labelled_shapes=False, use_cupy=False):
    'Create an image of shape with tetrahedrons and spheres in it'
    large_image_shape = np.multiply(shape, 2)
    large_image = np.zeros(large_image_shape)
    large_image2 = np.zeros(large_image_shape)

    mask = np.zeros(large_image_shape)
    mask2 = np.zeros(large_image_shape)

    quarter = large_image_shape[0]//4
    threequarter = large_image_shape[0] * 3 // 4
    xcoordinates = []
    ycoordinates = []
    
    centrecoordinates = []
    #for i in range(N_particles):
    n = 0
    attempts = 0
    while n < N_particles and attempts < 10:
        xysize = np.random.randint(minsize, maxsize)  # changed size from 32
        half_xysize = xysize // 2
        current_shape = 'None'

        random_float = np.random.random()

        label = {
            'tetrahedron': 1,
            'cube': 2,
            'ellipsoid': 3,
        }

        if random_float < fraction_tetra:
            current_shape = 'tetrahedron'
            ell, large_ell = generate_tetrahedron_flat(xysize, use_cupy=use_cupy)
            large_ell = ell # large_ell has wrong shape for rest of function
        elif random_float < (fraction_tetra + fraction_cube):
            # cube
            current_shape = 'cube'
            ell = generate_cuboid_flat(xysize)
            large_ell = ell
        else:
            current_shape = 'ellipsoid'
            ell, large_ell = generate_random_ellipsoid(xysize, also_make_with_higher_radii=True, higher_radii=higher_radii)
        
        xcoordinate = np.random.randint(quarter - half_xysize, threequarter-half_xysize)
        ycoordinate = np.random.randint(quarter - half_xysize, threequarter-half_xysize) #(0, large_image_shape[1] - xysize)
        centre_x = xcoordinate + xysize // 2 - quarter
        centre_y = ycoordinate + xysize // 2 - quarter

        crop = np.s_[xcoordinate:xcoordinate+xysize, ycoordinate:ycoordinate+xysize]
        
        if no_overlap:
            ell_contains_counts = ell.astype(bool)
            counts = large_image[crop][ell_contains_counts].sum()
            if attempts >= 10:
                break
            if counts > 0:
                attempts += 1
                continue
        attempts = 0
        large_image[crop] += ell
        centrecoordinates.append((centre_x, centre_y))

        large_image2[crop] += large_ell
        intmask = ell.astype(bool).astype("uint8")
        intmask2 = ell.astype(bool).astype("uint8")
        if labelled_shapes:
            mask[crop] += intmask * label[current_shape]
            mask2[crop] += intmask2 * label[current_shape]
        else:
            mask[crop] += intmask
            mask2[crop] += intmask2
        n += 1
    central_image = np.s_[quarter:threequarter, quarter:threequarter]
    image = large_image[central_image]
    image2 = large_image2[central_image]
    mask = mask[central_image]
    mask2 = mask2[central_image]
    return image, image2, mask, mask2, centrecoordinates


def old_ellipsoid_and_tetrahedron_image(N_particles=10, shape=(128, 128), higher_radii=2, minradius=6, fraction_tetra=1/3, fraction_cube=1/3, no_overlap=False, labelled_shapes=False, use_cupy=False):
    'Create an image of shape with tetrahedrons and spheres in it'
    large_image_shape = np.multiply(shape, 2)
    large_image = np.zeros(large_image_shape)
    large_image2 = np.zeros(large_image_shape)

    mask = np.zeros(large_image_shape)
    mask2 = np.zeros(large_image_shape)

    quarter = large_image_shape[0]//4
    threequarter = large_image_shape[0] * 3 // 4
    xcoordinates = []
    ycoordinates = []
    
    centrecoordinates = []
    #for i in range(N_particles):
    n = 0
    attempts = 0
    while n < N_particles and attempts < 10:
        xysize = np.random.randint(5,70)  # changed size from 32
        half_xysize = xysize // 2
        current_shape = 'None'

        random_float = np.random.random()

        label = {
            'tetrahedron': 1,
            'cube': 2,
            'ellipsoid': 3,
        }

        if random_float < fraction_tetra:
            current_shape = 'tetrahedron'
            ell, large_ell = generate_tetrahedron_flat(xysize, use_cupy=use_cupy)
            large_ell = ell # large_ell has wrong shape for rest of function
        elif random_float < (fraction_tetra + fraction_cube):
            # cube
            current_shape = 'cube'
            ell = generate_cuboid_flat(xysize)
            large_ell = ell
        else:
            current_shape = 'ellipsoid'
            ell, large_ell = generate_random_ellipsoid(xysize, also_make_with_higher_radii=True, higher_radii=higher_radii)
        radius = np.array([-1, -1])
        
        while (radius < minradius).any():
            xcoordinate = np.random.randint(quarter - half_xysize, threequarter-half_xysize)
            ycoordinate = np.random.randint(quarter - half_xysize, threequarter-half_xysize) #(0, large_image_shape[1] - xysize)
            centre_x = xcoordinate + xysize // 2 - quarter
            centre_y = ycoordinate + xysize // 2 - quarter

            radius = get_radius((centre_x, centre_y), centrecoordinates)

        
        crop = np.s_[xcoordinate:xcoordinate+xysize, ycoordinate:ycoordinate+xysize]
        
        if no_overlap:
            ell_contains_counts = ell.astype(bool)
            counts = large_image[crop][ell_contains_counts].sum()
            if attempts >= 10:
                break
            if counts > 0:
                attempts += 1
                continue
        attempts = 0
        large_image[crop] += ell
        centrecoordinates.append((centre_x, centre_y))

        large_image2[crop] += large_ell
        intmask = ell.astype(bool).astype("uint8")
        intmask2 = ell.astype(bool).astype("uint8")
        if labelled_shapes:
            mask[crop] += intmask * label[current_shape]
            mask2[crop] += intmask2 * label[current_shape]
        else:
            mask[crop] += intmask
            mask2[crop] += intmask2
        n += 1
    central_image = np.s_[quarter:threequarter, quarter:threequarter]
    image = large_image[central_image]
    image2 = large_image2[central_image]
    mask = mask[central_image]
    mask2 = mask2[central_image]
    return image, image2, mask, mask2, centrecoordinates


def normalise(img, max=1):
    img = img - img.min()
    img = img / img.max()
    img = max*img
    return img

def add_noise(img):
    img = img + np.random.random(size=img.shape)
    img = img + np.random.poisson(img)
    return img

def create_data(particle_number=20, shape=(128, 128), centre_labels=False, higher_radii=1, minsize=5, maxsize=40, fraction_tetra=1/3, fraction_cube=1/3, no_overlap=False, labelled_shapes=False):
    img, img2, mask, mask2, centres = ellipsoid_and_tetrahedron_image(particle_number, shape=shape, minsize=minsize, maxsize=maxsize, fraction_tetra=fraction_tetra, fraction_cube=fraction_cube, no_overlap=no_overlap, labelled_shapes = labelled_shapes, higher_radii=higher_radii)
    #particles = mask == 1
    #overlaps = mask2 == 2
    #particles[overlaps] = False
    #background = np.ones(img.shape)
    #background[np.logical_or(particles, overlaps)] = 0

    #particles = particles.astype("uint8")
    #overlaps = overlaps.astype("uint8")
    #background = background.astype("uint8")

    #label = np.stack([background, particles, overlaps], axis=-1)
    if centre_labels:
        label = np.ravel_multi_index(np.transpose(centres), img.shape) # USING CENTRES AS LABEL
        label = label.astype("uint16")
    else:
        label = mask
    img = add_noise(img)
    img = normalise(img)
    return img, label


def save_dataset_numpy(filename = "data", nImages=100, shape=(128, 128), minnumber=1, maxnumber=30, noise=True, cross=False, fraction_tetra=1/3, fraction_cube=1/3, no_overlap=False, labelled_shapes=False, save_example=True):
    'Draw images with tetrahedrons and ellipsoids on it. Returns raw, labels.'
    p = Path('dataset')
    p.mkdir(exist_ok=True, parents=True)

    labels = []
    raw = []
    for i in tqdm(range(nImages), desc = "Images"):
        N = np.random.randint(minnumber,maxnumber)
        img, label = create_data(particle_number=N, shape=shape, centre_labels=False, fraction_tetra=fraction_tetra, fraction_cube=fraction_cube, no_overlap=no_overlap, labelled_shapes = labelled_shapes,)
        raw.append(img)
        labels.append(label)
    #labels = np.array(labels, dtype="uint8"), 
    if cross:
        histograms = []
        for entry in labels:
            hist = np.zeros(np.prod(img.shape))
            hist[entry] = 1
            histograms.append(hist)
        labels = np.array(histograms)
    raw = np.array(raw)
    labels = np.array(labels)

    np.save(p / (filename + "_labels.npy"), labels)
    np.save(p / (filename + "_raw.npy"), raw)


    if save_example:
        imageio.imsave(p / (filename + "_label_0.png"), normalise(labels[0], max=255).astype(np.uint8))
        imageio.imsave(p / (filename + "_raw_0.png"), normalise(raw[0], max=255).astype(np.uint8))

    return raw, labels

import pickle

def save_dataset_pickle(filename = "data", overwrite=False, nImages=100, shape=(128, 128), minnumber=1, maxnumber=30, minsize=5, maxsize=40, noise=True, cross=False, fraction_tetra=1/3, fraction_cube=1/3, no_overlap=False, labelled_shapes=False, save_example=True):
    'Draw images with tetrahedrons and ellipsoids on it. Returns raw, labels.'
    p = Path('dataset')
    p.mkdir(exist_ok=True, parents=True)

    writemode = 'ab' if not overwrite else 'wb'
    i = 0
    try:
        with open(p / '{}.pickle'.format(filename), writemode) as f:
            for i in tqdm(range(nImages), desc = "Images"):
                N = np.random.randint(minnumber,maxnumber)
                img, label = create_data(
                    particle_number=N, shape=shape, centre_labels=False, 
                    minsize=minsize, maxsize=maxsize, 
                    fraction_tetra=fraction_tetra, fraction_cube=fraction_cube, 
                    no_overlap=no_overlap, labelled_shapes = labelled_shapes,)

                img = img[..., None]
                label = label[..., None]
                pickle.dump((img, label), f)
    except KeyboardInterrupt:
        print(f"User stopped after {i} writes")


def load_generator_pickle(filename='data'):
    p = Path('dataset')
    x = open(p / f'{filename}.pickle', 'rb')
    def generator():
        'Yields (X, y) dataset'
        while True:
            try:
                yield pickle.load(x)
            except EOFError:
                break
    return generator

def Tetrahedron(vertices):
    """
    Given a list of the xyz coordinates of the vertices of a tetrahedron, 
    return tetrahedron coordinate system
    """
    origin, *rest = vertices
    mat = (np.array(rest) - origin).T
    tetra = np.linalg.inv(mat)
    return tetra, origin

def pointInside_Tetrahedron(point, tetra, origin, use_cupy=False):
    """
    Takes a single point or array of points, as well as tetra and origin objects returned by 
    the Tetrahedron function.
    Returns a boolean or boolean array indicating whether the point is inside the tetrahedron.
    """
    if use_cupy:
        import cupy as np
    else:
        import numpy as np
    tetra = np.array(tetra)
    origin = np.array(origin)
    point = np.array(point)
    
    newp = np.matmul(tetra, (point-origin).T).T
    mask = np.all(newp>=0, axis=-1) & np.all(newp <=1, axis=-1) & (np.sum(newp, axis=-1) <=1)
    if use_cupy:
        return np.asnumpy(mask)
    return mask

def get_axes_minmax(vertices, num):
    axes = [np.linspace(x0, xf, num) for x0, xf in zip(vertices.min(0), vertices.max(0))]
    minmax = [(x0, xf) for x0, xf in zip(vertices.min(0), vertices.max(0))]
    return axes, minmax

def random_angle():
        return 360*np.random.random(())

def generate_tetrahedron(size=30, rotate=True, use_cupy=False):
    "rotate can be False"
    # Default standing-up tetrahedron
    vertices = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ])
    if rotate:
        if type(rotate) == tuple:
            rot_matrix = rotation_matrix(rotate)
        else:
            rot_matrix = rotation_matrix((random_angle(), random_angle(), random_angle()))
        new_vertices = (rot_matrix @ vertices.T).T
    else:
        new_vertices = vertices

    axes, minmax = get_axes_minmax(new_vertices, size)

    X, Y, Z = np.meshgrid(*axes)
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    tetra, origin = Tetrahedron(new_vertices)
    mask = pointInside_Tetrahedron(points, tetra, origin, use_cupy=use_cupy)

    mask = mask.reshape(X.shape)
    return mask

def generate_tetrahedron_flat(size=30, higher_radius=2, rotate=True, use_cupy=False):
    if rotate and not type(rotate) == tuple:
        rotate = (random_angle(), random_angle(), random_angle())
    tet1 = generate_tetrahedron(size=size, rotate=rotate, use_cupy=use_cupy).sum(-1)
    tet2 = generate_tetrahedron(size=size+higher_radius, rotate=rotate, use_cupy=use_cupy).sum(-1)
    return tet1, tet2

def generate_half_sizes(size):
    'Size is full width of volume'
    return np.random.randint(1, size//2-1)

def sensible_angle():
    angles = [0, 30, 45, 60, 90, 120, 135, 150]
    i = int(np.random.randint(0, len(angles)))
    return angles[i]

def generate_cuboid_old(volume_size, rotate=True, random=True):
    'Generates a rotated 3D cube. Size must be even.'
    # size should be even
    #if volume_size%2 == 1:
    #    volume_size +=1

    volume_shape = 3*(volume_size,) # should be odd
    volume = np.zeros(volume_shape)
    hwidth, hheight, hdepth = (generate_half_sizes(volume_shape[0]) for i in range(3))

    centre_of_volume = (volume_size)//2
    cov = centre_of_volume

    volume_slice = np.s_[cov - hwidth : cov + hwidth, cov - hheight : cov + hheight, cov - hdepth : cov + hdepth]

    volume[volume_slice] = 1
    if rotate:
        if random:
            a1, a2 = random_angle(), random_angle()
        else:
            a1, a2 = sensible_angle(), sensible_angle()
        volume = ndimagerotate(volume, a1, axes=(1,0))
        volume = ndimagerotate(volume, a2, axes=(2,1))

    return volume

def generate_cuboid_old_flat(size, rotate=True, random=True):
    return generate_cuboid_old(size, rotate, random).sum(-1)

def regular_cube_vertices_simple():
    '''Four vertices all closest to the first index of a regular cube,
    aligned with the coordinate system axes
    '''
    vertices = np.array(
        [[-1, -1, -1], # this one is origin
         [ 1, -1, -1],
         [-1,  1, -1],
         [-1, -1,  1]])
    return vertices

def regular_cube_vertices():
    ci = np.array([-1, -1, -1])
    cf = 0 - ci # symmetric points across zero
    cs = np.array([ci, cf])
    all_vertices = list(itertools.product(*cs.T))

    corner_vertices = all_vertices[:3] + [all_vertices[4]]
    return np.array(corner_vertices), np.array(all_vertices)


def rectangular_cuboid_vertices(corner_indices=[-1, -2, -3]):
    '''
    Four vertices all closest to the first index of a rectangular cuboid,
    aligned with the coordinate system axes
    '''
    ci = np.array(corner_indices)
    cf = 0 - ci # symmetric points across zero

    vertices = [ci]
    for i in range(3):
        c = ci.copy()
        c[i] = cf[i]
        vertices.append(c)
    return np.array(vertices)

def pointInsideCuboid(points, vertices):
    O, *ABC = vertices
    P = points
    A,B,C = ABC
    OA, OB, OC = ABC - O

    POA = P @ OA
    POB = P @ OB
    POC = P @ OC
    
    L1 = (O @ OA < POA) & (POA < A @ OA)
    L2 = (O @ OB < POB) & (POB < B @ OB) 
    L3 = (O @ OC < POC) & (POC < C @ OC)
    return (L1 & L2 & L3).T

def pointInsideCuboid2(points, vertices):
    O, *ABC = vertices
    P = points
    A,B,C = ABC
    
    OA, OB, OC = ABC - O
    AO, BO, CO = O - ABC
    OP = P - O
    
    AP = P - A
    BP = P - B
    CP = P - C
    
    L1 = OP@OA > 0
    L2 = OP@OB > 0
    L3 = OP@OC > 0
    L4 = AP@AO > 0
    L5 = BP@BO > 0
    L6 = CP@CO > 0
    
    return L1 & L2 & L3 & L4 & L5 & L6

def rotated_cuboid_vertices(angles=None):
    corner_vertices, all_vertices = regular_cube_vertices()
    if angles:
        rot_matrix = rotation_matrix(angles)
    else:
        rot_matrix = rotation_matrix((random_angle(), random_angle(), random_angle()))
    rot_corner_vertices = (rot_matrix @ corner_vertices.T).T
    rot_all_vertices = (rot_matrix @ all_vertices.T).T
    return rot_corner_vertices, rot_all_vertices

def generate_cuboid(size, angles=None):
    rot_corner_vertices, rot_all_vertices = rotated_cuboid_vertices(angles)
    
    sqrt3 = 3**0.5
    x = np.linspace(-sqrt3, sqrt3, size)

    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    points = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
    V = pointInsideCuboid(points, rot_corner_vertices).reshape(X.shape).astype("uint32")
    return V

def generate_cuboid_flat(size, angles=None):
    return generate_cuboid(size, angles=angles).sum(-1)