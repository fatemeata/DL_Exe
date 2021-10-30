import os.path
import json
import pickle

import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size  # the batch size
        self.image_size = image_size  # the image size
        self.rotation = rotation  # flags for different augmentations and whether the data should be shuffled for each epoch
        self.mirroring = mirroring  # flags for mirroring
        self.shuffle = shuffle  # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        # TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        images = []
        labels = []
        # TODO: implement next method

        json_file = open(self.label_path)
        data = json.load(json_file)

        images = []
        labels = []
        rand_list = random.sample(range(0, len(data)), self.batch_size)
        for i in rand_list:
            images.append(os.path.join(self.file_path, str(i) + '.' + 'npy'))
            labels.append(data.get(str(i)))

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict.get((x))

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        print(self.class_name(1))
        images, labels = self.next()
        fig, axs = plt.subplots(nrows=4, ncols=3)
        for i, ax in enumerate(fig.axes):
            with open(images[i], 'rb') as file:
                my_image = np.load(file)
                ax.imshow(my_image)
                ax.set_title(self.class_name(labels[i]))
                ax.axis('off')

        plt.show()
        return


img = ImageGenerator("data\exercise_data", "data\labels.json", 12, 128)

img.show()
