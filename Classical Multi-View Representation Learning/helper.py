"""
This file contains helper functions which can be used across the project.
"""
import random
import numpy as np


def rank(N: int, type_embedding, img_embeds, rec_embeds):
    """
    This function is used to calculate Median rank and recall@k
    :param N: Number of samples to take
    :param type_embedding: whether "image" or "text"
    :param img_embeds: image embeddings
    :param rec_embeds: text embeddings
    :return:
    """

    NN = img_embeds.shape[0]
    im_vecs = img_embeds
    instr_vecs = rec_embeds

    # Ranker
    idxs = range(N)

    glob_rank = []

    # Calculate recall for k=1,5,10
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(10):

        ids = random.sample(range(0, NN), N)
        im_sub = im_vecs[ids, :]
        instr_sub = instr_vecs[ids, :]

        if type_embedding == 'image':
            sims = np.dot(im_sub, instr_sub.T)  # for im2recipe
        else:
            sims = np.dot(instr_sub, im_sub.T)  # for recipe2im

        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        for ii in idxs:
            # get a column of similarities
            sim = sims[ii, :]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1

            # store the position
            med_rank.append(pos + 1)

        for i in recall.keys():
            recall[i] = recall[i] / N

        med = np.median(med_rank)

        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10

    return np.average(glob_rank), glob_recall
