import tensorflow as tf
from .generate import load_generator_pickle
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def num_batches(total_items, batch_size):
    return int(total_items / batch_size)

def get_generator(g, gaussian_kernel=1, normalise=True, binary_labels=False):
    def gen():
        while True:
            try:
                a, b = g.__next__()
                if normalise:
                    a -= a.min()
                    a /= a.max()
                if binary_labels:
                    b[b > 0] = 1
                if gfilter:
                    a = gaussian_filter(a, (gaussian_kernel, gaussian_kernel, 0))
                yield a, b
            except:
                return
    return gen

def dataset_function(dataset='data', epochs=10, batch_size=32, gaussian_kernel=1, normalise=True, binary_labels=False):
    print(f"Epochs: {epochs}\nBatch Size: {batch_size}")
    shapes = (128, 128, 1), (128, 128, 1)
    dataset = tf.data.Dataset.from_generator(
        generator=get_generator(
            load_generator_pickle(dataset)(), 
            gaussian_kernel=gaussian_kernel, 
            normalise=normalise, 
            binary_labels=binary_labels
        ),
        output_types=(tf.float64, tf.float64),
        output_shapes=shapes)
    dataset = dataset.batch(batch_size).repeat(epochs)
    return dataset

def plot(*imgs):
    fig, AX = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        AX[0, i].imshow(img)

def plot_from_tf_dataset(dataset, i=0):
    img, lab = dataset.as_numpy_iterator().__next__()
    plot(img[i,...,0], lab[i,...,0])