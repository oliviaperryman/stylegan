import os
import pickle
import numpy as np
import PIL.Image
from PIL import Image
from tqdm import tqdm

import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc

import matplotlib.pyplot as plt

dir = 'results/vm/landscapes-no-cond/'
fn = 'network-snapshot-006126.pkl'
rnd = np.random.RandomState(1)

# Conditioning
num_classes = 1
names = "Landscapes"
# names = ["Romanticism", "Impressionism", "Realism", "Post-Impressionism"]

# Initialize TensorFlow
tflib.init_tf()

_G, _D, Gs = pickle.load(open(os.path.join(dir, fn), 'rb'))

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


def main():
    view10()
    # save_n(1000)
    # view_through_mapping()
    # get_dlatents()
    # directions = ['water', 'grass', 'outdoor', 'field', 'beach', 'lake', 'ocean', 'painting', 'river', 'tree',
    #               'nature', 'night', 'sunset', 'wave', 'blue', 'mountain', 'clouds', 'hill']

    # transform_image(directions)


def transform_image(labels):
    qlatents = rnd.randn(1, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(qlatents, None)

    for label in labels:
        direction = pickle.load(
            open(dir + 'generated_imgs1000/directions/'+label+'.p', 'rb'))
        move_and_show(label, dlatents[0], direction, [-5, -2, 0, 2, 5])

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, latent_vector.shape[0], 512))
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)
    images = Gs.components.synthesis.run(
        latent_vector, randomize_noise=False, **synthesis_kwargs)
    img = PIL.Image.fromarray(images[0], 'RGB')
    return img.resize((256, 256))


def move_and_show(title,latent_vector, direction, coeffs):
    fig, ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    fig.suptitle(title,fontsize=16)
    plt.show()


def get_dlatents():
    qlatents = pickle.load(
        open(dir+'generated_imgs1000/landscape_latents.p', 'rb'))
    d_latents = []
    for q in tqdm(qlatents):
        dlatents = Gs.components.mapping.run(np.array([q]), None)
        d_latents.append(dlatents[0])
    pickle.dump(d_latents, open(
        dir+'generated_imgs1000/landscape_dlatents.p', 'wb'))


def view_through_mapping():
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)

    qlatents = np.random.normal(size=(1, Gs.input_shape[1]))

    dlatents = Gs.components.mapping.run(qlatents, None)
    images = Gs.components.synthesis.run(
        dlatents, randomize_noise=False, **synthesis_kwargs)

    images2 = Gs.run(qlatents, None, randomize_noise=False,
                     output_transform=fmt)

    for im in images:
        img = PIL.Image.fromarray(im, 'RGB')
        plt.imshow(img, interpolation='bicubic')
        plt.axis('off')
        plt.show()


def save_n(n, style=None):
    all_latents = []

    for i in tqdm(range(n)):
        # Generate image.
        latents = rnd.randn(1, Gs.input_shape[1])
        all_latents.append(latents[0])
        # c = np.tile(conditioning[style],(1,1))
        images = Gs.run(latents, None, truncation_psi=0.7,
                        randomize_noise=True, output_transform=fmt)

        # Save image.
        png_filename = dir + 'generated_imgs1000/img_{}.png'.format(i)
        # png_filename = dir + "/"+ names[style]+ '/img_{}.png'.format(i)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        pickle.dump(all_latents, open(
            dir + "generated_imgs1000/landscape_latents.p", "wb"))
        # pickle.dump( all_latents, open(  dir + "/"+ names[style] + "/landscape_latents.p", "wb" ))


def view10():
    # Initialize conditioning
    conditioning = None  # np.eye(num_classes)

    rnd = np.random.RandomState(0)

    # Pick latent vector.
    latents = rnd.randn(10, Gs.input_shape[1])

    for i in range(num_classes):
        # Generate image.
        #c = np.tile(conditioning[i],(10,1))
        images = Gs.run(latents, conditioning, truncation_psi=0.7,
                        randomize_noise=True, output_transform=fmt)

        plt.figure(1, figsize=(30, 3))
        print(names[i])

        for im in range(len(images)):
            plt.subplot(1, 10, im+1)
            img = PIL.Image.fromarray(images[im], 'RGB')
            plt.imshow(img, interpolation='bicubic')
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
