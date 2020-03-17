import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator

import matplotlib.pyplot as plt


def generate_image(Gs, latents):
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7,
                    randomize_noise=True, output_transform=fmt)
    result = []
    for im in images:
        x = np.squeeze(im)
        img = PIL.Image.fromarray(x, 'L')
        result.append(img)
    return result


def main():
    tflib.init_tf()
    model_path = "./results/vm/network-snapshot-002364.pkl"

    with open(model_path, "rb") as f:
        _G, _D, Gs = pickle.load(f)

    rnd = np.random.RandomState(42)

    # Pick random latent vectors.
    latents = rnd.randn(5, Gs.input_shape[1])

    imgs = generate_image(Gs, latents)

    for i in range(1, len(imgs)):
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(imgs[0], cmap='gray')
        fig.add_subplot(1, 2, 2)
        plt.imshow(imgs[i], cmap='gray')
        plt.title("Euclidean distance: " +
                  str(np.linalg.norm(latents[0]-latents[i])))
        plt.show()


if __name__ == "__main__":
    main()
