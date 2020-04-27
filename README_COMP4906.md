# COMP 4906 Honours Thesis

## Data
- landscapes/landscapes
  - all images after preprocessing
- landscapes/train
  - all wikiart images with no processing


## Code files

- landscapes/preprocess-landscapes.ipynb
  - Select images with 4 desired styles
  - Crop and resize all landscape images to 512
  - Save the style label with the filename
- analyze_images.py
  - Use Microsoft Vision API to label 1000 generated images 
- distances.py 
  - Measures and displays the euclidean distance between latent vectors of generated images (mnist dataset)
- learn_direction.py
  - Uses Logistic Regression and pre-labelled data to find the direction of certain features in the latent space.
  - based on existing file "Learn_direction_in_latent_space.ipynb"
- mixing_latents.py
  - Mixes two latent variables to produce new images
- subjective-test.py
  - Present user with various images and transformations and record responses to Likert-style questions
- landscapes/vgg-landscapes.ipynb
  - Using the pretrained VGG network (trained on Imagenet) the existing and generated images are labelled
- landscapes/visualizations
  - Files needed to use the Embedding Projector 
  - Visualization of 100 generated landscapes in embedding space


## Modified files

- dataset_tool.py
  - read in landscape images and style labels 
- encode_images.py
  - Find the latent vector of an image for a particular pretrained StyleGAN model.
- train.py
  - Add settings for mnist, cifar and landscapes training
- training/training_loop.py
  - Add checkpoint to resume training