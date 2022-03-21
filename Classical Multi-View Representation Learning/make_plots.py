"""
This file is useful to make plots of median ranks and recall@k
"""

import matplotlib.pyplot as plt
import pickle


def make_plot(final_list: list, title: str):
    latent_dims_list = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700]
    xi = list(range(len(latent_dims_list)))
    labels = ['all', 'title', 'ingredients', 'instructions']
    for final_val, label in zip(final_list, labels):
        plt.plot(xi, final_val, marker='o', linestyle='--', label=label)
    plt.xticks(xi, latent_dims_list)
    plt.xlabel('latent dimensions')
    plt.ylabel('value')
    plt.title(title)
    plt.legend()
    plt.savefig('./results/images/' + title + '.png')
    plt.show()


filename_list = ['./results/prof_val_cca_all_text_1k.pkl', './results/prof_val_cca_title_text_1k.pkl',
                 './results/prof_val_cca_ingredients_text_1k.pkl', './results/prof_val_cca_instructions_text_1k.pkl']

final_medr_list = list()
final_recall_1_list = list()
final_recall_5_list = list()
final_recall_10_list = list()

for filename in filename_list:
    with open(filename, 'rb') as f:
        dct = pickle.load(f)
        final_medr_list.append(dct['medr'])
        recall_1_list = list()
        recall_5_list = list()
        recall_10_list = list()
        for element in dct['recall_k']:
            recall_1_list.append(element[1.0])
            recall_5_list.append(element[5.0])
            recall_10_list.append(element[10.0])
        final_recall_1_list.append(recall_1_list)
        final_recall_5_list.append(recall_5_list)
        final_recall_10_list.append(recall_10_list)

make_plot(final_medr_list, 'Comparison plot; Median Rank; 1k text to image')
make_plot(final_recall_1_list, 'Comparison plot; Recall@1; 1k text to image')
make_plot(final_recall_5_list, 'Comparison plot; Recall@5; 1k text to image')
make_plot(final_recall_10_list, 'Comparison plot; Recall@10; 1k text to image')
