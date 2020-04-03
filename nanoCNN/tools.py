import numpy as np
import imageio

def one_hot_to_grayscale(fit):
    return np.argmax(fit, axis=-1)#np.where(np.equal(fit.T, fit.max(-1).T).T)[-1].reshape((len(fit), 128, 128))

def segment(dataset):
    dataset = dataset.copy()
    background = dataset < 0
    particles = dataset >= 0
    dataset[background] = 0
    dataset[particles] = 1
    return dataset

def normalise(img):
    img = img.copy()
    img = img - img.min(axis=(-2,-3))[:,None,None]
    img = img / img.max(axis=(-2,-3))[:,None,None]
    return img

def split_image_into_128(img):
    arr = np.array(np.split(img, 16))
    arr = np.array(np.split(arr, 16, -1))
    arr = arr.reshape((-1, 128,128))[..., None]
    return arr

def segment_reshape_tiled_img(fit):
    img = segment(fit)
    img = reshape_tiled_img(img)
    return img

def reshape_tiled_img(img):
    img = img.reshape((16, 16, 128, 128))
    img = np.concatenate(img, -1)
    img = np.concatenate(img, -2)
    return img
def save_png_true_fit(name, img, fit):
    img = img / img.max()
    imageio.imwrite(name + "_true.png", (255*img).astype('uint8'))
    imageio.imwrite(name + "_fit.png", (255*fit).astype('uint8'))

def predict_images(raw, model, split_retile=False):
    if split_retile:
        raw = split_image_into_128(raw)
    raw = normalise(raw)
    fit = model.predict(raw)
    fit = one_hot_to_grayscale(fit)
    if split_retile:
        fit = reshape_tiled_img(fit)
    return fit
