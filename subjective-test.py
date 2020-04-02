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

num = 8
rnd = np.random.RandomState(num)

# Initialize TensorFlow
tflib.init_tf()
_G, _D, Gs = pickle.load(open(os.path.join(dir, fn), 'rb'))

directions = ['water', 'grass', 'outdoor', 'field', 'beach', 'lake', 'ocean', 'painting', 'river', 'tree',
              'nature', 'night', 'sunset', 'wave', 'blue', 'mountain', 'clouds', 'hill']

direction_vectors = {x: pickle.load(
    open(dir + 'generated_imgs1000/directions/'+x+'.p', 'rb')) for x in directions}


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
    return img  # .resize((256, 256))


def move_and_show(title, latent_vector, direction, coeffs):
    fig, ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    fig.suptitle(title, fontsize=16)
    plt.show()


def add_direction(direction, coeff, latent_vector):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return new_latent_vector


def run_experiment(show_both):
    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(qlatents, None)

    for latent in dlatents:
        result = {}
        if(show_both):
            plt.imshow(generate_image(latent))
            plt.axis('off')
            plt.show()

        coeff = 0
        new_latent = latent.copy()
        coeff = int(rnd.random()*10)
        new_latent = add_direction(
            direction_vectors['tree'], coeff, new_latent)

        plt.imshow(generate_image(new_latent))
        plt.axis('off')
        plt.show()

        is_landscape = input("Is landscape (y/n)?")

        is_tree = input(
            "I can recognize tree(s) in the lansdcape? \n 1: Strongly Disagree \n 2: Disagree \n 3: Neutral \n 4: Agree \n 5: Strongly Agree\n")

        # result['qlatents'] = qlatents
        # result['dlatents'] = dlatents
        result['dlatent'] = latent
        result['coeff'] = coeff
        result["is_landscape"] = is_landscape
        result["is_tree"] = is_tree
        results.append(result)
    return results

def run_experiment2():
    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(qlatents, None)

    for latent in dlatents:
        result = {}
        x = rnd.random_integers(0,1,1)[0]
        
        plt.subplot(1,2,1 if x else 2)
        plt.imshow(generate_image(latent))
        plt.title("1" if x else "2")
        plt.axis('off')
        

        coeff = 0
        new_latent = latent.copy()
        coeff = int(rnd.random()*10)
        new_latent = add_direction(
            direction_vectors['tree'], coeff, new_latent)

        plt.subplot(1,2,2 if x else 1)
        plt.imshow(generate_image(new_latent))
        plt.title("2" if x else "1")
        plt.axis('off')
        plt.show()

        rating = input("Which image is more treelike? 1 or 2? 0 if neither: ")

        result['order'] = x
        result['dlatent'] = latent
        result['coeff'] = coeff
        result['rating'] = rating
        results.append(result)
    return results

def run_random_direction():
    random_direction = rnd.randn(16,512)
    
    results = []
    qlatents = rnd.randn(10, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(qlatents, None)

    new_latent = dlatents[0]

    new_latents = [new_latent]
    for i in range(1,10):
        new_latent = add_direction(
            random_direction, i, dlatents[0])
        new_latents.append(new_latent)

    plt.figure(1, figsize=(30, 3))
    for i, dlatent in enumerate(new_latents):
        plt.subplot(1, 10, i+1)
        im = generate_image(dlatent)
        plt.imshow(im)
        plt.axis('off')
    # plt.savefig(exp_dir+filename[:-2]+".png")
    plt.show()


def run_random_direction_q():
    random_direction = rnd.randn(512)
    
    qlatents = rnd.randn(1, Gs.input_shape[1])

    new_latent = qlatents[0]

    new_latents = [new_latent]
    for i in range(1,10):
        new_latent = qlatents[0] + i * random_direction
        new_latents.append(new_latent)

    dlatents = Gs.components.mapping.run(np.array(new_latents), None)

    plt.figure(1, figsize=(30, 3))
    for i, dlatent in enumerate(dlatents):
        plt.subplot(1, 10, i+1)
        im = generate_image(dlatent)
        plt.imshow(im)
        plt.axis('off')
    # plt.savefig(exp_dir+filename[:-2]+".png")
    plt.show()

def run_test(filename, show_both=True):
    exp_dir = dir+'experiments/'

    results = run_experiment(show_both)

    pickle.dump(results, open(exp_dir+filename, 'wb'))

def run_test2(filename):
    exp_dir = dir+'experiments/'

    results = run_experiment2()

    pickle.dump(results, open(exp_dir+filename, 'wb'))


def view_results(filename):
    exp_dir = dir+'experiments/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    results = sorted(results, key=(lambda x: x["coeff"]), reverse=False)

    filtered_results = list(
        filter(lambda x: x["is_landscape"] == 'y', results))

    x = [int(r["coeff"]) for r in results]
    y = [int(r["is_tree"]) for r in results]

    plt.plot(x, y)
    plt.xlabel('Coefficient towards tree')
    plt.ylabel('Agreement')
    plt.xlim(0, 10)
    plt.ylim(1, 5)
    plt.show()

    x = [int(r["coeff"]) for r in filtered_results]
    y = [int(r["is_tree"]) for r in filtered_results]

    plt.plot(x, y)
    plt.xlabel('Coefficient towards tree')
    plt.ylabel('Agreement')
    plt.xlim(0, 10)
    plt.ylim(1, 5)
    plt.title('Tree-ness with non-landscapes filtered out')
    plt.show()


def view_results2(filename):
    exp_dir = dir+'experiments/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    results = sorted(results, key=(lambda x: x["coeff"]), reverse=False)

    x = [int(r["coeff"]) for r in results]
    y = []
    for r in results:
        if(r['rating'] == '0'):
            y.append(0)
        elif(r['order'] == 1 and r['rating'] == '1'):
            y.append(-1)
        elif(r['order'] == 1 and r['rating'] == '2'):
            y.append(1)
        if(r['order'] == 0 and r['rating'] == '1'):
            y.append(1)
        elif(r['order'] == 0 and r['rating'] == '2'):
            y.append(-1)

    plt.scatter(x, y)
    plt.xlabel('Coefficient towards tree')
    plt.ylabel('Chose more tree')
    plt.xlim(0, 10)
    plt.ylim(-1,1)
    plt.title('Tree-ness compared side by side')
    plt.show()


def view_avg_results():
    exp_dir = dir+'experiments/'
    x = []
    y = []

    for filename in listdir(exp_dir):
        if('show_one' in filename and 'png' not in filename):
            results = pickle.load(open(exp_dir+filename, 'rb'))
            for r in results:
                if r["is_landscape"] == "y":
                    x.append(int(r["coeff"]))
                    y.append(int(r["is_tree"]))

    plt.scatter(x, y)
    plt.xlabel('Coefficient towards tree')
    plt.ylabel('Agreement')
    plt.xlim(0, 10)
    plt.ylim(1, 5)
    plt.title('Tree-ness with non-landscapes filtered out')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.show()


def view_imgs(filename):
    exp_dir = dir+'experiments/'
    results = pickle.load(open(exp_dir+filename, 'rb'))
    results = sorted(results, key=(lambda x: x["coeff"]), reverse=False)
    # filtered_results = list(
    #     filter(lambda x: x["is_landscape"] == 'y', results))

    plt.figure(1, figsize=(30, 3))
    for i, r in enumerate(results):
        plt.subplot(2, 10, i+1)
        im = generate_image(r['dlatent'])
        plt.imshow(im)
        plt.axis('off')
        plt.subplot(2, 10, 10 + i+1)
        new_latent = add_direction(
            direction_vectors['tree'], r['coeff'], r['dlatent'])
        im = generate_image(new_latent)
        plt.imshow(im)
        plt.title('Coeff: ' + str(r['coeff']))
        plt.axis('off')
    plt.savefig(exp_dir+filename[:-2]+".png")
    plt.show()


def fix_results(filename):
    exp_dir = dir+'experiments/'
    results = pickle.load(open(exp_dir+filename, 'rb'))

    new_results = []
    for i, r in enumerate(results):
        r['dlatent'] = r['dlatents'][i]
        new_results.append(r)

    pickle.dump(results, open(exp_dir+filename, 'wb'))



def main():
    # filename = "olivia/results" + str(num) + ".p"
    # filename = "results-show_one.p"
    # run_test(filename, show_both=False)
    # view_imgs(filename)
    # view_results(filename)
    # view_avg_results()

    # run_test2(filename)
    # view_imgs(filename)
    # view_results2(filename)
    for _ in range(3):
        run_random_direction_q()


if __name__ == "__main__":
    main()
