from PIL import Image as pil_image
import numpy as np
import cv2
from data_utility import image_normalization
from keras.utils import Sequence
import keras.backend as K

def load_img(path, target_size):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = image_normalization(img)
    return img

class DataGenerator(Sequence):
    """Class for image data iterators."""
    def __init__(self, data_info, batch_size=32, img_cols=170, 
        img_rows=170, num_au_labels=12, shuffle=True):

        self.n = len(data_info)
        self.batch_size = batch_size
        self.target_size = (img_cols, img_rows)
        self.num_labels = num_au_labels
        self.shuffle = shuffle
        self.info = data_info
        self.on_epoch_end()
    
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
    
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx},'
                             'but the Sequence '
                             'has length {length}'.format(idx=idx, 
                                                          length=len(self)))
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_samples(index_array)

    def _get_batches_of_samples(self, index_array):
        """Gets a batch of samples.


        # Arguments
            index_array: array of sample indices to include in batch.


        # Returns
            A batch of samples.
        """
        batch_x = np.zeros(shape=(len(index_array), self.target_size[0],
            self.target_size[1], 3), dtype=K.floatx())
        batch_y = np.zeros(shape=(len(index_array), self.num_labels), 
            dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = load_img(self.info[j][0], self.target_size)
            batch_x[i] = x
            batch_y[i] = self.info[j][1]
        return batch_x, batch_y     
    
    def on_epoch_end(self):
        self._set_index_array()

    
    def __len__(self):
        return (self.n + self.batch_size - 1) //self.batch_size


def load_batch_from_sets(data_sets, img_cols, img_rows):
    
    num_au_labels = len(data_sets[0][1])
    n = len(data_sets)
    image_batch = np.zeros(shape=(n, img_cols, img_rows, 3), dtype=np.float32)
    label_batch = np.zeros((n, num_au_labels), dtype=np.float32)

    b = 0
    while b < n:
        
        image_batch[b] = load_img(data_sets[b][0], (img_cols, img_rows))
        label_batch[b] = data_sets[b][1]

        b += 1
    
    return image_batch, label_batch


