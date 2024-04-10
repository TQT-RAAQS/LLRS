import os
from pathlib import Path

import dill as pickle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as Img


class Image:
    '''
    Common image class to save meta information and the image buffer

    Brooke Dolny
    Winter 2021

    Sailesh Bechar
    Fall 2020
    '''

    def __init__(self, meta_information, image_buffer):
        self.meta_information = meta_information
        self.image_buffer = image_buffer

    def display_most_recent_image(self):
        '''
        Displays the first image in image buffer saved in class
        '''
        print("Displaying images")
        name = (
            "ampIndex"
            + str(self.meta_information.amp_index)
            + "preampIndex"
            + str(self.meta_information.preamp_gain_index)
            + "_"
            + self.meta_information.get_hs_speed()
        )
        plt.figure(name, figsize=(12, 8))
        plt.title(name)
        plt.imshow(self.image_buffer[:, :, 0])
        plt.show()

    def save_all_images(self, folder, filename):
        """
        Saves each image in the image buffer as a 16-bit greyscale PNG in the specifed folder
        with the prefix filename.

        Parameters
        ----------
        folder
            The folder to save the image in.
        filename
            The file prefix to use. Each image will use this as the filename with a integer + '.png'
            suffix to differentiate the saved image files.
        """
        shape = np.shape(self.image_buffer)
        Path(folder).mkdir(parents=True, exist_ok=True)
        for i in range(shape[2]):
            im = Img.fromarray(self.image_buffer[:, :, i], 'I')  # or more verbose as Image.fromarray(ar, 'I;16')
            im.save(os.path.join(folder, filename + "_" + str(i) + ".png"))

    def save_most_recent_image(self, folder, filename):
        """
        Saves the most recent image in the image buffer as a 16-bit greyscale PNG. The image is
        saved in the specifed folder with the spacified filename.

        Parameters
        ----------
        folder
            The folder to save the image in. If the folder does not exist, it is created.
        filename
            The filename to save the image with, without the prefix '.png'.
        """
        shape = np.shape(self.image_buffer)
        im = Img.fromarray(self.image_buffer[:, :, shape[2] - 1], 'I')  # or more verbose as Image.fromarray(ar, 'I;16')
        Path(folder).mkdir(parents=True, exist_ok=True)
        im.save(os.path.join(folder, filename + ".png"))

    def save_pickle(self, folder, file_name):
        '''
        Saves image object as a pickle file to a specified path. Creates the specified
        folder if it does not exist.

        Parameters
        ----------
        folder(str) : folder of destination path
        file_name(str) : name of where to store file
        '''
        # Write image object to a pickle file
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(folder, file_name + '.pickle'), 'wb') as fi:
            # dump your data into the file
            pickle.dump(self, fi)

    @staticmethod
    def load(file_name):
        '''
        Loads image object from a pickle file at a specified path

        Parameters
        ----------
        file_name(str) : folder of destination path
        '''
        with open(file_name, "rb") as fi:
            return pickle.load(fi)
