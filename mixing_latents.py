import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import glob
import tensorflow as tf

import matplotlib.pyplot as plt
from tqdm import tqdm

from encoder.generator_model import Generator


dir = 'results/vm/landscapes-no-cond/'
fn = 'network-snapshot-006126.pkl'
fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


num = 0
rnd = np.random.RandomState(num)

tflib.init_tf()
_G, _D, Gs = pickle.load(open(os.path.join(dir,fn), 'rb'))

generator = Generator(Gs, batch_size=1, randomize_noise=False, model_res=512)

def load_latents():
    encoded_latents = []
    latents_dir = "./latent_representations/landscapes-no-cond/"
    for filename in os.listdir(latents_dir):
        latent = np.load(latents_dir + filename)
        encoded_latents.append(latent)
    return encoded_latents

def generate_image(latent):
    images = Gs.run(np.array([latent]), None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    img = PIL.Image.fromarray(images[0], 'RGB')
    return img

def generate_image2(latent_vector):
    latent_vector = latent_vector.reshape((1, 16, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

def view(latents):
    for i, latent in enumerate(latents):
        plt.subplot(1,10,i+1)
        im = generate_image2(latent)
        plt.imshow(im)
        plt.axis('off')
    plt.show()

def view_mixed(latent1,latent2, func=generate_image2):
    latents = []
    n=4

    for i in range(n+1):
        proportional_latents = []
        proportional_latents.extend([latent1] * (n-i))
        proportional_latents.extend([latent2] * i)
        new_latent = np.mean(np.array(proportional_latents), axis=0)
        latents.append(new_latent)

    for i, latent in enumerate(latents):
        plt.subplot(1,n+1,i+1)
        im = func(latent)
        plt.imshow(im)
        plt.axis('off')
    plt.show()

def main():
    latents = load_latents()
    #view(latents)
    view_mixed(latents[2],latents[6])

    rnd = np.random.RandomState(0)
    latents = rnd.randn(10, Gs.input_shape[1])
    
    view_mixed(latents[2],latents[3], func=generate_image)
    view_mixed(latents[2],latents[5], func=generate_image)
    



if __name__ == "__main__":
    main()