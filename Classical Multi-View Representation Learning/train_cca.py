import pickle
import numpy as np
from datetime import datetime

from cca_zoo.models import CCA
from constants import TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH, VALIDATION_IMAGE_PATH, VALIDATION_TEXT_PATH
from helper import rank


def print_current_time():
    print("Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def create_dataset(filepath: str):

    print('Fetching data from file', filepath)
    f = open(filepath, 'rb')
    data = None

    while True:
        try:
            tmp = pickle.load(f)
            if data is None:
                data = tmp
            else:
                data = np.vstack((data, tmp))
        except EOFError:
            break

    print('Completed fetching data from file', filepath)
    return data


print_current_time()
image_train_data = create_dataset(TRAIN_IMAGE_PATH)
print_current_time()
text_train_data = create_dataset(TRAIN_TEXT_PATH)

print('Train image shape', image_train_data.shape)
print('Train text shape', text_train_data.shape)

print_current_time()
image_val_data = create_dataset(VALIDATION_IMAGE_PATH)
print_current_time()
text_val_data = create_dataset(VALIDATION_TEXT_PATH)

print('Validation image shape', image_val_data.shape)
print('Validation text shape', text_val_data.shape)

cca = CCA(latent_dims=100)
# cca = CCA(n_components=100, algorithm="svd")
# cca = linear_cca()

print('Starting Training')
print_current_time()
cca.fit((image_train_data, text_train_data))
print_current_time()
print('Ending Training')

print('Start transforming and calculating ranks')
image_data_c, text_data_c = cca.transform((image_val_data, text_val_data))
medr, recall_k = rank(1000, "image", image_data_c, text_data_c)

print('Median', medr)
print('Recall', recall_k)
# image_data_c, text_data_c = cca.test(image_data, text_data)

print_current_time()
print('End transforming and calculating ranks')
