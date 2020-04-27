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

num = 102
rnd = np.random.RandomState(num)

# Initialize TensorFlow
if 0:
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
                      qlatents2[i] * (9-coeffs[i])) / 9

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

        result = Result(qlatents[i], np.mean(
            dlatents[1], axis=0), direction, coeffs[i])
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
    plt.plot(np.unique(x), np.poly1d(np.polyfit(
        x, y_novelty_agreement, 1))(np.unique(x)))
    plt.show()


def view_novelty_results_summary(filename, exp_num):
    exp_dir = 'experiments/experiment'+str(exp_num)+'/novelty/'

    filenames = os.listdir(exp_dir)

    results = []
    for fn in filenames:
        rs = pickle.load(open(exp_dir+fn, 'rb'))
        results.extend(rs)

    # Sort in order of distance
    results = sorted(results, key=(lambda x: x.distance), reverse=False)

    x = [int(r.distance) for r in results]
    y_novelty_agreement = [int(r.novel_agreement) for r in results]

    plt.scatter(x, y_novelty_agreement)
    plt.title("Experiment " + str(exp_num))
    plt.xlabel('Distance from original image')
    plt.ylabel('Agreement with novelty')
    # line of best fit
    plt.plot(np.unique(x), np.poly1d(np.polyfit(
        x, y_novelty_agreement, 1))(np.unique(x)))
    plt.show()


def run_experiment1_quality(filename):
    exp_dir = 'experiments/experiment1/quality/'
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

        x = rnd.random_integers(0, 1, 1)[0]

        plt.subplot(1, 2, 1 if x else 2)
        plt.imshow(generate_image(dlatents[0]))
        plt.title("1" if x else "2")
        plt.axis('off')

        plt.subplot(1, 2, 2 if x else 1)
        plt.imshow(generate_image(dlatents[1]))
        plt.title("2" if x else "1")
        plt.axis('off')
        plt.show()

        landscapelike = input(
            "Which image more resembles a landscape? 1 or 2? 0 if neither: ")
        treelike = input(
            "Which image more resembles a tree or trees? 1 or 2? 0 if neither: ")
        oceanlike = input(
            "Which image more resembles an ocean? 1 or 2? 0 if neither: ")

        result = Result(qlatents[i], new_latent, random_direction, coeffs[i])
        labels = {'tree': treelike, 'ocean': oceanlike}
        result.quality_result(landscapelike, labels, x)

        print(result)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def run_experiment2_quality(filename):
    exp_dir = 'experiments/experiment2/quality/'
    # Create target Directory if don't exist
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

        x = rnd.random_integers(0, 1, 1)[0]

        plt.subplot(1, 2, 1 if x else 2)
        plt.imshow(generate_image(dlatents[0]))
        plt.title("1" if x else "2")
        plt.axis('off')

        plt.subplot(1, 2, 2 if x else 1)
        plt.imshow(generate_image(dlatents[2]))
        plt.title("2" if x else "1")
        plt.axis('off')
        plt.show()

        landscapelike = input(
            "Which image more resembles a landscape? 1 or 2? 0 if neither: ")
        treelike = input(
            "Which image more resembles a tree or trees? 1 or 2? 0 if neither: ")
        oceanlike = input(
            "Which image more resembles an ocean? 1 or 2? 0 if neither: ")

        result = Result(qlatents[i], new_latent, None,
                        coeffs[i], orig_latent2=qlatents2[i])
        labels = {'tree': treelike, 'ocean': oceanlike}
        result.quality_result(landscapelike, labels, x)

        print(result)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def run_experiment3_quality(filename):
    exp_dir = 'experiments/experiment3/quality/'
    # Create target Directory if don't exist
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(qlatents, None)

    for latent in dlatents:
        x = rnd.random_integers(0, 1, 1)[0]

        direction_label = rnd.choice(['tree', 'ocean'])

        plt.subplot(1, 2, 1 if x else 2)
        plt.imshow(generate_image(latent))
        plt.title("1" if x else "2")
        plt.axis('off')

        new_latent = latent.copy()
        coeff = int(rnd.random()*10)
        new_latent = add_direction(
            direction_vectors[direction_label], coeff, new_latent)

        plt.subplot(1, 2, 2 if x else 1)
        plt.imshow(generate_image(new_latent))
        plt.title("2" if x else "1")
        plt.axis('off')
        plt.show()

        landscapelike = input(
            "Which image more resembles a landscape? 1 or 2? 0 if neither: ")
        treelike = input(
            "Which image more resembles a tree or trees? 1 or 2? 0 if neither: ")
        oceanlike = input(
            "Which image more resembles an ocean? 1 or 2? 0 if neither: ")

        result = Result(latent, new_latent, direction_label, coeff)
        labels = {'tree': treelike, 'ocean': oceanlike}
        result.quality_result(landscapelike, labels, x)

        print(result)

        results.append(result)

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def view_quality_results(filename, exp_num, label_names):
    exp_dir = 'experiments/experiment'+str(exp_num)+'/quality/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    results = sorted(results, key=(lambda x: x.coeff), reverse=False)

    x = [int(r.distance) for r in results]
    y = []
    for r in results:
        print(r)
        if(r.landscape == '0'):
            y.append(0)
        elif(r.order == 1 and r.landscape == '1'):
            y.append(-1)
        elif(r.order == 1 and r.landscape == '2'):
            y.append(1)
        elif(r.order == 0 and r.landscape == '1'):
            y.append(1)
        elif(r.order == 0 and r.landscape == '2'):
            y.append(-1)

    plt.scatter(x, y)
    plt.xlabel('Coefficient')
    plt.ylabel('Chose "More resembles a landscape"')
    plt.ylim(-1, 1)
    plt.title('Landscape resemblance compared side by side')
    plt.show()

    for label in label_names:
        filtered_results = list(
            filter(lambda x: x.direction == label, results))
        x = [int(r.distance) for r in filtered_results]
        y = []
        for r in filtered_results:
            if(r.labels[label] == '0'):
                y.append(0)
            elif(r.order == 1 and r.labels[label] == '1'):
                y.append(-1)
            elif(r.order == 1 and r.labels[label] == '2'):
                y.append(1)
            elif(r.order == 0 and r.labels[label] == '1'):
                y.append(1)
            elif(r.order == 0 and r.labels[label] == '2'):
                y.append(-1)
        plt.scatter(x, y)
        plt.xlabel('Coefficient towards ' + label)
        plt.ylabel('Chose more ' + label)
        plt.ylim(-1, 1)
        plt.title(label + ' direction compared side by side')
        plt.show()


def view_quality_results_bar(filename, exp_num, label_names):
    exp_dir = 'experiments/experiment'+str(exp_num)+'/quality/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    results = sorted(results, key=(lambda x: x.coeff), reverse=False)

    x = [int(r.coeff) for r in results]
    moved_img_is_more = np.zeros(len(x))
    none_more = np.zeros(len(x))
    orig_img_is_more = np.zeros(len(x))

    for i, r in enumerate(results):
        if(r.landscape == '0'):
            none_more[i] = 1
        elif(r.order == 1 and r.landscape == '1'):
            orig_img_is_more[i] = 1
        elif(r.order == 1 and r.landscape == '2'):
            moved_img_is_more[i] = 1
        elif(r.order == 0 and r.landscape == '1'):
            moved_img_is_more[i] = 1
        elif(r.order == 0 and r.landscape == '2'):
            orig_img_is_more[i] = 1

    ax = plt.subplot(111)
    red = ax.bar(x, orig_img_is_more, width=0.2, color='r', align='center')
    blue = ax.bar(x, none_more, width=0.2, color='b', align='center')
    green = ax.bar(x, moved_img_is_more, width=0.2, color='g', align='center')
    plt.xlabel('Coefficient')
    plt.ylabel('Number of times chosen')
    plt.title('Landscape resemblance compared side by side')
    plt.legend([green, red, blue],  ["The moved image more resembles a landscape.",
                                     "The original image more resembles a landscape.", "Neither image more resembles a landscape."])
    plt.show()

    for label in label_names:
        filtered_results = list(
            filter(lambda x: x.direction == label, results))
        x = [int(r.distance) for r in filtered_results]
        y = []
        for r in filtered_results:
            if(r.labels[label] == '0'):
                y.append(0)
            elif(r.order == 1 and r.labels[label] == '1'):
                y.append(-1)
            elif(r.order == 1 and r.labels[label] == '2'):
                y.append(1)
            elif(r.order == 0 and r.labels[label] == '1'):
                y.append(1)
            elif(r.order == 0 and r.labels[label] == '2'):
                y.append(-1)
        plt.scatter(x, y)
        plt.xlabel('Coefficient towards ' + label)
        plt.ylabel('Chose more ' + label)
        plt.ylim(-1, 1)
        plt.title(label + ' direction compared side by side')
        plt.show()


def view_quality_results_bar_summary(exp_num, label_names):
    exp_dir = 'experiments/experiment'+str(exp_num)+'/quality/'

    filenames = os.listdir(exp_dir)

    results = []
    for fn in filenames:
        rs = pickle.load(open(exp_dir+fn, 'rb'))
        results.extend(rs)

    results = sorted(results, key=(lambda x: x.coeff), reverse=False)

    x = np.arange(10)
    moved_img_is_more = np.zeros(10)
    none_more = np.zeros(10)
    orig_img_is_more = np.zeros(10)

    for r in results:
        if(r.landscape == '0'):
            none_more[r.coeff] += 1
        elif(r.order == 1 and r.landscape == '1'):
            orig_img_is_more[r.coeff] += 1
        elif(r.order == 1 and r.landscape == '2'):
            moved_img_is_more[r.coeff] += 1
        elif(r.order == 0 and r.landscape == '1'):
            moved_img_is_more[r.coeff] += 1
        elif(r.order == 0 and r.landscape == '2'):
            orig_img_is_more[r.coeff] += 1

    ax = plt.subplot(211)
    red = ax.bar(np.array(x)-0.2, orig_img_is_more,
                 width=0.2, color='r', align='center')
    blue = ax.bar(x, none_more, width=0.2, color='b', align='center')
    green = ax.bar(np.array(x)+0.2, moved_img_is_more,
                   width=0.2, color='g', align='center')
    plt.xlabel('Coefficient')
    plt.ylabel('Number of times chosen')
    plt.title('Landscape resemblance compared side by side')
    plt.subplot(212)
    plt.axis('off')
    plt.tight_layout()
    plt.legend([green, red, blue],  ["The moved image more resembles a landscape.",
                                     "The original image more resembles a landscape.", "Neither image more resembles a landscape."],
               )
    plt.show()

    for label in label_names:
        x = np.arange(10)
        moved_img_is_more = np.zeros(10)
        none_more = np.zeros(10)
        orig_img_is_more = np.zeros(10)

        filtered_results = list(
            filter(lambda x: x.direction == label, results))

        for r in filtered_results:
            if(r.landscape == '0'):
                none_more[r.coeff] += 1
            elif(r.order == 1 and r.landscape == '1'):
                orig_img_is_more[r.coeff] += 1
            elif(r.order == 1 and r.landscape == '2'):
                moved_img_is_more[r.coeff] += 1
            elif(r.order == 0 and r.landscape == '1'):
                moved_img_is_more[r.coeff] += 1
            elif(r.order == 0 and r.landscape == '2'):
                orig_img_is_more[r.coeff] += 1

        ax = plt.subplot(211)
        red = ax.bar(np.array(x)-0.2, orig_img_is_more,
                     width=0.2, color='r', align='center')
        blue = ax.bar(x, none_more, width=0.2, color='b', align='center')
        green = ax.bar(np.array(x)+0.2, moved_img_is_more,
                       width=0.2, color='g', align='center')
        plt.xlabel('Coefficient')
        plt.ylabel('Number of times chosen')
        plt.title(label + ' resemblance compared side by side')
        plt.subplot(212)
        plt.axis('off')
        plt.tight_layout()
        plt.legend([green, red, blue],  ["The moved image more resembles a " + label,
                                         "The original image more resembles a " + label, "Neither image more resembles a " + label])
        plt.show()


def main():
    filename = "olivia" + str(num) + ".p"

    # run_experiment1_novelty(filename)
    # view_novelty_results_summary(filename, 1)
    # run_experiment2_novelty(filename)
    # view_novelty_results_summary(filename, 2)
    # run_experiment3_novelty(filename)
    # view_novelty_results_summary(filename, 3)

    # run_experiment1_quality(filename)
    # view_quality_results_bar(filename, 1, [])
    view_quality_results_bar_summary(1, [])

    # run_experiment2_quality(filename)
    # view_quality_results_bar(filename, 2, [])
    view_quality_results_bar_summary(2, [])

    # run_experiment3_quality(filename)
    # view_quality_results(filename, 3, ['tree', 'ocean'])
    # view_quality_results_bar(filename, 3, ['tree', 'ocean'])
    view_quality_results_bar_summary(3, ['tree', 'ocean'])


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

    def quality_result(self, landscape, labels, order):
        self.landscape = landscape
        self.labels = labels
        self.order = order

    def __str__(self):
        string = "Result. "
        string += str(self.orig_latent) + "\n"
        string += str(self.moved_latent) + "\n"
        string += str(self.direction) + "\n"
        string += str(self.coeff) + "\n"
        string += str(self.landscape) + "\n"
        string += str(self.labels) + "\n"
        string += str(self.order) + "\n"
        return string


if __name__ == "__main__":
    main()
