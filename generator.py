import os.path
import json
import pickle

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import math


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
        self.shuffle = shuffle
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        json_file = open(label_path)
        self.label_data = json.load(json_file)
        self.dataset_size = len(self.label_data)
        self.total_batch = math.ceil(self.dataset_size / self.batch_size)
        self.add_last = False
        if self.total_batch * self.batch_size > self.dataset_size:
            self.add_last = True
        # self.image_data =
        self.current_batch = 0
        self.curr_epoch = 0
        self.data_index = np.arange(0, self.dataset_size)
        json_file.close()

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

        images = []
        labels = []

        if self.current_batch == 0 and self.shuffle:  # check whether it's the first batch or not
            np.random.shuffle(self.data_index)

        if self.current_batch == self.total_batch-1 and self.add_last:
            pr_images = self.current_batch * self.batch_size
            remain = self.dataset_size - pr_images
            l1 = self.data_index[(self.current_batch * self.batch_size):]
            l2 = self.data_index[0:self.batch_size - remain]
            current_list = np.concatenate((l1, l2), axis=0)

        else:
            current_list = self.data_index[self.current_batch * self.batch_size: (self.current_batch+1) * self.batch_size]  ## get index of images corresponding to the current batch


        for index in current_list:
            image = np.load(os.path.join(self.file_path, str(index) + '.' + 'npy'))
            aug_image = self.augment(image)
            res_image = resize(aug_image, (self.image_size[0], self.image_size[1]))
            images.append(res_image)
            labels.append(self.label_data.get(str(index)))

        if self.current_batch == self.total_batch:
            print("END OF THE EPOCH #", self.curr_epoch)
            self.curr_epoch += 1
            self.current_batch = 0
        else:
            self.current_batch += 1
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        degree_list = [90, 180, 270]
        if self.rotation:
            rotation_p = np.random.random()
            rand_degree = np.random.choice(degree_list)
            if rotation_p > 0.25:
                img = ndimage.rotate(img, rand_degree, reshape=False)

        if self.mirroring:
            mirroring_p = np.random.random()
            if mirroring_p > 0.5:
                img = np.flip(img)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.curr_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict.get((x))

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        images, labels = self.next()
        n_img = len(images)
        fig, axs = plt.subplots(nrows=int(np.ceil(n_img/3)), ncols=3)

        for i in range(n_img):
            ax = axs.flatten()[i]
            ax.imshow(images[i])
            ax.set_title(self.class_name(labels[i]))
            ax.axis('off')
            if (self.batch_size % 3 == 1):
                axs.flat[-1].set_visible(False)
                axs.flat[-2].set_visible(False)
            if (self.batch_size % 3 == 2):
                axs.flat[-1].set_visible(False)
        plt.show()

        # for i, ax in enumerate(fig.axes):
        #     ax.imshow(images[i])
        #     ax.set_title(self.class_name(labels[i]))
        #     ax.axis('off')
        # plt.show()
        return


img = ImageGenerator("data\exercise_data", "data\labels.json", 10, [32, 32, 3],
                     rotation=False, mirroring=True, shuffle=False)
img.show()

