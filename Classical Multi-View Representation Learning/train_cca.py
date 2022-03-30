"""
This file is useful to train linear CCA on recipe1M dataset.
"""
import pickle
import numpy as np

from datetime import datetime
from cca_zoo.models import CCA

from constants import TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH, VALIDATION_IMAGE_PATH, VALIDATION_TEXT_PATH, TEST_IMAGE_PATH,\
    TEST_TEXT_PATH, TEXT_ELEMENT, TYPE_EMBEDDING
from helper import rank


def print_current_time():
    print("Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def create_dataset(filepath: str):
    """
    load dataset from pickle file
    :param filepath: path of the file
    :return: numpy array
    """
    print('Fetching data from file', filepath)
    f = open(filepath, 'rb')
    data = list()

    while True:
        try:
            # data.append(pickle.load(f))
            return pickle.load(f)
        except EOFError:
            break

    print('Completed fetching data from file', filepath)
    return np.vstack(data)


def compute_rank_val_test(cca, image_data, text_data):
    """
    Compute median rank and recall@k for validation and test set.
    :param cca: CCA object
    :param image_data: image embeddings
    :param text_data: text embeddings
    :return: ranks for test and validation
    """
    print('Start transforming and calculating ranks')
    if TYPE_EMBEDDING == "image":
        image_data_c, text_data_c = cca.transform((image_data, text_data))
    else:
        text_data_c, image_data_c = cca.transform((text_data, image_data))

    medr_1k, recall_k_1k = rank(1000, TYPE_EMBEDDING, image_data_c, text_data_c)
    print('search pool 1k latent dimensions', latent_dims, 'Median ', medr_1k)
    print('search pool 1k latent dimensions', latent_dims, 'Recall', recall_k_1k)

    medr_10k, recall_k_10k = rank(10000, TYPE_EMBEDDING, image_data_c, text_data_c)
    print('search pool 10k latent dimensions', latent_dims, 'Median ', medr_10k)
    print('search pool 10k latent dimensions', latent_dims, 'Recall', recall_k_10k)

    print_current_time()
    print('End transforming and calculating ranks')

    return medr_1k, recall_k_1k, medr_10k, recall_k_10k


def compute_rank(latent_dims: int):
    """
    Train CCA and compute ranks for test and validation
    :param latent_dims: latent dimension
    :return:
    """

    cca = CCA(latent_dims=latent_dims)
    print('Starting Training for', latent_dims)
    print_current_time()
    if TYPE_EMBEDDING == "image":
        cca.fit((image_train_data, text_train_data))
    else:
        cca.fit((text_train_data, image_train_data))
    print('Ending Training for', latent_dims)
    print_current_time()

    print('Validation')
    medr_1k_val, recall_k_1k_val, medr_10k_val, recall_k_10k_val = compute_rank_val_test(cca, image_val_data,
                                                                                         text_val_data)
    print('Test')
    medr_1k_test, recall_k_1k_test, medr_10k_test, recall_k_10k_test = compute_rank_val_test(cca, image_test_data,
                                                                                             text_test_data)

    return medr_1k_val, recall_k_1k_val, medr_10k_val, recall_k_10k_val, medr_1k_test, recall_k_1k_test, medr_10k_test,\
           recall_k_10k_test


if __name__ == "__main__":

    print_current_time()
    # image_train_data = create_dataset(TRAIN_IMAGE_PATH)
    # print_current_time()
    # text_train_data = create_dataset(TRAIN_TEXT_PATH)
    data = create_dataset(TRAIN_IMAGE_PATH)
    image_train_data = data[0]
    if TEXT_ELEMENT == "all":
        text_train_data = data[1]
    else:
        text_train_data = create_dataset(TRAIN_TEXT_PATH)[0]

    print('Train image shape', image_train_data.shape)
    print('Train text shape', text_train_data.shape)

    # print_current_time()
    # image_val_data = create_dataset(VALIDATION_IMAGE_PATH)
    # print_current_time()
    # text_val_data = create_dataset(VALIDATION_TEXT_PATH)

    data = create_dataset(VALIDATION_IMAGE_PATH)
    image_val_data = data[0]
    if TEXT_ELEMENT == "all":
        text_val_data = data[1]
    else:
        text_val_data = create_dataset(VALIDATION_TEXT_PATH)[0]

    print('Validation image shape', image_val_data.shape)
    print('Validation text shape', text_val_data.shape)

    # print_current_time()
    # image_test_data = create_dataset(TEST_IMAGE_PATH)
    # print_current_time()
    # text_test_data = create_dataset(TEST_TEXT_PATH)

    data = create_dataset(TEST_IMAGE_PATH)
    image_test_data = data[0]
    if TEXT_ELEMENT == "all":
        text_test_data = data[1]
    else:
        text_test_data = create_dataset(TEST_TEXT_PATH)[0]

    print('Test image shape', image_test_data.shape)
    print('Test text shape', text_test_data.shape)

    # n_cores = int(mp.cpu_count())
    # print('Number of cores', n_cores)
    # p = mp.Pool(processes=1)

    latent_dims_list = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700]

    medr_1k_val_list = list()
    recall_k_1k_val_list = list()
    medr_10k_val_list = list()
    recall_k_10k_val_list = list()

    medr_1k_test_list = list()
    recall_k_1k_test_list = list()
    medr_10k_test_list = list()
    recall_k_10k_test_list = list()

    for latent_dims in latent_dims_list:
        result = compute_rank(latent_dims)
        medr_1k_val_list.append(result[0])
        recall_k_1k_val_list.append(result[1])
        medr_10k_val_list.append(result[2])
        recall_k_10k_val_list.append(result[3])

        medr_1k_test_list.append(result[4])
        recall_k_1k_test_list.append(result[5])
        medr_10k_test_list.append(result[6])
        recall_k_10k_test_list.append(result[7])

    with open('./results/prof_val_cca_' + TEXT_ELEMENT + "_" + TYPE_EMBEDDING + '_1k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_1k_val_list, 'recall_k': recall_k_1k_val_list, 'latent_dims': latent_dims_list}, f)

    with open('./results/prof_val_cca_' + TEXT_ELEMENT + "_" + TYPE_EMBEDDING + '_10k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_10k_val_list, 'recall_k': recall_k_10k_val_list, 'latent_dims': latent_dims_list}, f)

    with open('./results/prof_test_cca_' + TEXT_ELEMENT + "_" + TYPE_EMBEDDING + '_1k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_1k_test_list, 'recall_k': recall_k_1k_test_list, 'latent_dims': latent_dims_list}, f)

    with open('./results/prof_test_cca_' + TEXT_ELEMENT + "_" + TYPE_EMBEDDING + '_10k' + '.pkl', 'wb') as f:
        pickle.dump({'medr': medr_10k_test_list, 'recall_k': recall_k_10k_test_list, 'latent_dims': latent_dims_list}, f)


