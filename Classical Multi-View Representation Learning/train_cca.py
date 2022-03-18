import pickle
import numpy as np

from datetime import datetime

from cca_zoo.models import CCA
from constants import TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH, VALIDATION_IMAGE_PATH, VALIDATION_TEXT_PATH, TEXT_ELEMENT,\
    TYPE_EMBEDDING
from helper import rank


def print_current_time():
    print("Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def create_dataset(filepath: str):

    print('Fetching data from file', filepath)
    f = open(filepath, 'rb')
    data = list()

    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

    print('Completed fetching data from file', filepath)
    return np.vstack(data)


def compute_rank(latent_dims: int):

    cca = CCA(latent_dims=latent_dims)
    print('Starting Training for', latent_dims)
    print_current_time()
    if TYPE_EMBEDDING == "image":
        cca.fit((image_train_data, text_train_data))
    else:
        cca.fit((text_train_data, image_train_data))
    print('Ending Training for', latent_dims)
    print_current_time()

    print('Start transforming and calculating ranks')
    if TYPE_EMBEDDING == "image":
        image_data_c, text_data_c = cca.transform((image_val_data, text_val_data))
    else:
        text_data_c, image_data_c = cca.transform((text_val_data, image_val_data))

    medr_1k, recall_k_1k = rank(1000, TYPE_EMBEDDING, image_data_c, text_data_c)
    print('search pool 1k latent dimensions', latent_dims, 'Median ', medr_1k)
    print('search pool 1k latent dimensions', latent_dims, 'Recall', recall_k_1k)

    medr_10k, recall_k_10k = rank(10000, TYPE_EMBEDDING, image_data_c, text_data_c)
    print('search pool 10k latent dimensions', latent_dims, 'Median ', medr_10k)
    print('search pool 10k latent dimensions', latent_dims, 'Recall', recall_k_10k)

    print_current_time()
    print('End transforming and calculating ranks')

    return medr_1k, recall_k_1k, medr_10k, recall_k_10k


if __name__ == "__main__":

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

    # n_cores = int(mp.cpu_count())
    # print('Number of cores', n_cores)
    # p = mp.Pool(processes=1)

    latent_dims_list = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700]
    # final_list = [[image_train_data, image_val_data, text_train_data, text_val_data, x]
    #               for x in range(MIN_LATENT_DIMENSION, MAX_LATENT_DIMENSION+1)]
    # final_list = range(1, MAX_LATENT_DIMENSION + 1)
    # results = p.map(compute_rank, final_list)

    medr_1k_list = list()
    recall_k_1k_list = list()
    medr_10k_list = list()
    recall_k_10k_list = list()

    for latent_dims in latent_dims_list:
        result = compute_rank(latent_dims)
        medr_1k_list.append(result[0])
        recall_k_1k_list.append(result[1])
        medr_10k_list.append(result[2])
        recall_k_10k_list.append(result[3])

    print(medr_1k_list)
    print(recall_k_1k_list)
    print(medr_10k_list)
    print(recall_k_10k_list)

    with open('./results/cca_' + TEXT_ELEMENT + '_1k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_1k_list, 'recall_k': recall_k_1k_list, 'latent_dims': latent_dims_list}, f)

    with open('./results/cca_' + TEXT_ELEMENT + '_10k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_10k_list, 'recall_k': recall_k_10k_list, 'latent_dims': latent_dims_list}, f)


