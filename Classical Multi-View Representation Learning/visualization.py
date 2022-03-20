import pickle
import numpy as np

from cca_zoo.models import CCA
from constants import TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH


def create_dataset(filepath: str):

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


data = create_dataset(TRAIN_IMAGE_PATH)
image_train_data = data[0]
ids = data[2]
id_to_ind = dict()

for ind in range(len(ids)):
    id_to_ind[ids[ind]] = ind

text_train_data = create_dataset(TRAIN_TEXT_PATH)[0]

cca = CCA(latent_dims=500, random_state=0)
cca.fit((text_train_data, image_train_data))
print("Training is completed")
print("Transforming")


text_train_r, img_train_r = cca.transform((text_train_data, image_train_data))
# img_test_r, text_test_r = cca.transform((img_test, text_test))

# "title": "Chicken Lasagna", "id": "547c1e56aa"      // train
# "title": "Lasagna", "id": "4b12be6889"              // train
# "title": "Salad", "id": "ad0c7fc652"                // train
# "title": "Chicken Salad", "id": "4033b570e0"        // train

required_ids = ["547c1e56aa", "4b12be6889", "ad0c7fc652", "4033b570e0"]

required_test_embedding = text_train_r[id_to_ind[required_ids[0]]] - text_train_r[id_to_ind[required_ids[1]]] + \
                          text_train_r[id_to_ind[required_ids[2]]]

sims = np.dot(required_test_embedding, img_train_r.T)
sorting = np.argsort(sims)[::-1].tolist()

print("Indexes:", sorting[:10])
print("Expected index:", id_to_ind[required_ids[2]])
