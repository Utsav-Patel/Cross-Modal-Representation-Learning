"""
This file is useful to plot latent dimension data by reducing dimension using TSNE.
"""

import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from constants import TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH


def create_dataset(filepath: str):
    print('Fetching data from file', filepath)
    f = open(filepath, 'rb')
    data = list()

    while True:
        try:
            return pickle.load(f)
        except EOFError:
            break

    print('Completed fetching data from file', filepath)
    return np.vstack(data)


# Load cca model
with open('model.pkl', 'rb') as f:
    cca = pickle.load(f)

# Load train dataset
data = create_dataset(TRAIN_IMAGE_PATH)
image_train_data = data[0]
ids = data[2]
id_to_ind = dict()

for ind in range(len(ids)):
    id_to_ind[ids[ind]] = ind

print("image shape", data[0].shape)
text_train_data = create_dataset(TRAIN_TEXT_PATH)[0]

# Load ids of muffins and salads
with open('./muffin_ids.txt') as f:
    muffin_ids = [x.strip() for x in f.read().split('\n') if x.strip()]

with open('./salad_ids.txt') as f:
    salad_ids = [x.strip() for x in f.read().split('\n') if x.strip()]

muffin_text = list()
muffin_image = list()
salad_text = list()
salad_image = list()

# Fetch image and text corresponding embeddings for muffin and salad.
ignored_muffin_ids = 0
for muffin_id in muffin_ids:
    try:
        muffin_text.append(text_train_data[id_to_ind[muffin_id]])
        muffin_image.append(image_train_data[id_to_ind[muffin_id]])
    except:
        ignored_muffin_ids += 1

print('Total muffin ids', len(muffin_ids), 'Ignored:', ignored_muffin_ids)

ignored_salad_ids = 0
for salad_id in salad_ids:
    try:
        salad_text.append(text_train_data[id_to_ind[salad_id]])
        salad_image.append(image_train_data[id_to_ind[salad_id]])
    except:
        ignored_salad_ids += 1

print('Total salad ids', len(salad_ids), 'Ignored:', ignored_salad_ids)

# Choose number of points to select for dimensionality reduction for each category
num_points = 100
muffin_text_t, muffin_image_t = cca.transform((muffin_text[:num_points], muffin_image[:num_points]))
salad_text_t, salad_image_t = cca.transform((salad_text[:num_points], salad_image[:num_points]))

# Stack all numpy array
final_input_data = np.vstack([muffin_text_t, muffin_image_t, salad_text_t, salad_image_t])

# Initialize Tsne
tsne = TSNE(perplexity=150, n_jobs=-1)

print('Training TSNE')

# Fit and transform tsne
final_output_data = tsne.fit_transform(final_input_data)
print('Ending TSNE')

output_labels = [0] * num_points + [1] * num_points + [2] * num_points + [3] * num_points
legends = ['muffin_text', 'muffin_image', 'salad_text', 'salad_image']

# Final scatter plot
for i in range(0, 4 * num_points, num_points):
    plt.scatter(final_output_data[i:i + num_points, 0], final_output_data[i:i + num_points, 1],
                label=legends[i // num_points])
plt.legend()
plt.title('TSNE results: 100 points of each category')
plt.show()
