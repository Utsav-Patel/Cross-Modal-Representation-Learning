import pickle
import numpy as np
from datetime import datetime

# from sklearn.cross_decomposition import CCA
from cca_zoo.models import CCA


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
image_data = create_dataset('../data/image_features/image_train_embeddings.pkl')
print_current_time()
text_data = create_dataset('../data/embeddings_mean/text_title_train_embeddings.pkl')

print('Input image shape', image_data.shape)
print('Input text shape', text_data.shape)

cca = CCA(latent_dims=100)

print('Starting Training')
print_current_time()
cca.fit((image_data, text_data))
print('Ending Training')
print('Start transforming')
image_data_c, text_data_c = cca.transform((image_data, text_data))
print(image_data_c)
print(text_data_c)
print(image_data_c.shape)
print(text_data_c.shape)
print_current_time()
print('End transforming')
