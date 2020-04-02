import os
from os import listdir
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

num = 100
rnd = np.random.RandomState(num)

# Initialize TensorFlow
tflib.init_tf()
_G, _D, Gs = pickle.load(open(os.path.join(dir, fn), 'rb'))

directions = ['water', 'grass', 'outdoor', 'field', 'beach', 'lake', 'ocean', 'painting', 'river', 'tree',
              'nature', 'night', 'sunset', 'wave', 'blue', 'mountain', 'clouds', 'hill']

direction_vectors = {x: pickle.load(
    open(dir + 'generated_imgs1000/directions/'+x+'.p', 'rb')) for x in directions}


def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, latent_vector.shape[0], 512))
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)
    images = Gs.components.synthesis.run(
        latent_vector, randomize_noise=False, **synthesis_kwargs)
    img = PIL.Image.fromarray(images[0], 'RGB')
    return img


def generate_image_from_qlatent(latent_vector):
    dlatents = Gs.components.mapping.run(np.array([latent_vector]), None)

    return generate_image(dlatents[0])


def add_direction(direction, coeff, latent_vector):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return new_latent_vector


def run_experiment1_novelty(filename):
    exp_dir = 'experiments/experiment1/novelty/'
    # Create target Directory if don't exist
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])

    # Ensure there is one image at each distance from 0 to 9
    coeffs = np.random.permutation(10)

    for i in range(10):
        random_direction = rnd.randn(512)
        new_latent = qlatents[i] + coeffs[i] * random_direction

        latents = []
        latents.append(qlatents[i])
        latents.append(new_latent)

        dlatents = Gs.components.mapping.run(np.array(latents), None)

        plt.subplot(1, 2, 1)
        plt.imshow(generate_image(dlatents[0]))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(generate_image(dlatents[1]))
        plt.axis('off')
        plt.show()

        novel = input(
            "Is the second image novel compared to the first image? That is, is it significantly different? \n")

        agreement = input(
            "How much do you agree that the image is different?\n 1: Strongly Disagree \n 2: Disagree \n 3: Neutral \n 4: Agree \n 5: Strongly Agree\n")

        result = Result(qlatents[i], new_latent, random_direction, coeffs[i])
        result.novelty_result(novel, agreement)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def run_experiment2_novelty(filename):
    exp_dir = 'experiments/experiment2/novelty/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])
    qlatents2 = rnd.randn(10, Gs.input_shape[1])

    coeffs = np.random.permutation(10)

    for i in range(10):
        new_latent = (qlatents[i] * coeffs[i] +
                      qlatents2[i] * (9-coeffs[i])) / 2

        latents = []
        latents.append(qlatents[i])
        latents.append(qlatents2[i])
        latents.append(new_latent)

        dlatents = Gs.components.mapping.run(np.array(latents), None)

        plt.subplot(1, 3, 1)
        plt.imshow(generate_image(dlatents[0]))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(generate_image(dlatents[1]))
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(generate_image(dlatents[2]))
        plt.axis('off')
        plt.show()

        novel = input(
            "Is the third image novel compared to the first two images? That is, is it significantly different? \n")

        agreement = input(
            "How much do you agree that the image is different?\n 1: Strongly Disagree \n 2: Disagree \n 3: Neutral \n 4: Agree \n 5: Strongly Agree\n")

        result = Result(qlatents[i], new_latent, None, coeffs[i], qlatents2[i])
        result.novelty_result(novel, agreement)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def run_experiment3_novelty(filename):
    exp_dir = 'experiments/experiment3/novelty/'
    # Create target Directory if don't exist
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])

    # Ensure there is one image at each distance from 0 to 9
    coeffs = np.random.permutation(10)

    for i in range(10):
        direction = rnd.choice(['tree', 'ocean'])
        new_latent = qlatents[i]

        latents = []
        latents.append(qlatents[i])
        latents.append(new_latent)

        dlatents = Gs.components.mapping.run(np.array(latents), None)

        dlatents[1] = add_direction(
            direction_vectors[direction], coeffs[i], dlatents[1])

        plt.subplot(1, 2, 1)
        plt.imshow(generate_image(dlatents[0]))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(generate_image(dlatents[1]))
        plt.axis('off')
        plt.show()

        novel = input(
            "Is the second image novel compared to the first image? That is, is it significantly different? \n")

        agreement = input(
            "How much do you agree that the image is different?\n 1: Strongly Disagree \n 2: Disagree \n 3: Neutral \n 4: Agree \n 5: Strongly Agree\n")

        result = Result(qlatents[i], np.mean(dlatents[1],axis=0), direction, coeffs[i])
        result.novelty_result(novel, agreement)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def view_novelty_results(filename, exp_num):
    exp_dir = 'experiments/experiment'+str(exp_num)+'/novelty/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    # Sort in order of distance
    results = sorted(results, key=(lambda x: x.distance), reverse=False)

    x = [int(r.distance) for r in results]
    y_novelty_agreement = [int(r.novel_agreement) for r in results]

    plt.scatter(x, y_novelty_agreement)
    plt.title("Experiment " + str(exp_num))
    plt.xlabel('Distance from original image')
    plt.ylabel('Agreement with novelty')
    # line of best fit
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y_novelty_agreement, 1))(np.unique(x)))
    plt.show()


def main():
    filename = "olivia" + str(num) + ".p"

    # run_experiment1_novelty(filename)
    view_novelty_results(filename, 1)
    # run_experiment2_novelty(filename)
    view_novelty_results(filename, 2)
    # run_experiment3_novelty(filename)
    view_novelty_results(filename, 3)
    # run_experiment1_quality(filename)
    # run_experiment2_quality(filename)
    # run_experiment3_quality(filename)


class Result:
    def __init__(self, orig_latent, moved_latent, direction, coeff, orig_latent2=None):
        self.orig_latent = orig_latent
        self.moved_latent = moved_latent
        self.direction = direction
        self.coeff = coeff
        if(orig_latent2 is not None):
            self.distance = min(np.linalg.norm(
                orig_latent-moved_latent), np.linalg.norm(orig_latent2-moved_latent))
        else:
            self.distance = np.linalg.norm(orig_latent-moved_latent)

    def novelty_result(self, is_novel, novel_agreement):
        self.is_novel = is_novel
        self.novel_agreement = novel_agreement

    def quality_result(self, landscape, tree, ocean):
        self.landscape = landscape
        self.tree = tree
        self.ocean = ocean


if __name__ == "__main__":
    main()
