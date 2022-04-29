import torch
import random
import numpy as np
from time import time


def get_transformer_input(image_features, text_embedding, input_attention_mask):
    num_negative_to_positive_sample_ratio = 2

    input_batch_size = image_features.shape[0]
    output_batch_size = (num_negative_to_positive_sample_ratio + 1) * input_batch_size
    ground_truths = torch.zeros(output_batch_size)
    ground_truths[:input_batch_size] = 1

    final_image_features = torch.zeros(output_batch_size, *image_features.shape[1:])
    final_text_embeddings = torch.zeros(output_batch_size, *text_embedding.shape[1:])
    output_attention_mask = torch.zeros(output_batch_size, input_attention_mask.shape[1:])

    final_image_features[:input_batch_size] = image_features
    final_text_embeddings[:input_batch_size] = text_embedding

    for run_num in range(num_negative_to_positive_sample_ratio):
        a = torch.randperm(input_batch_size)
        b = torch.zeros(input_batch_size).to(dtype=torch.int64)
        for ind in range(input_batch_size):
            c = random.randint(0, input_batch_size - 1)
            while c == a[ind]:
                c = random.randint(0, input_batch_size - 1)
            b[ind] = c

        final_image_features[(1 + run_num) * input_batch_size : (2 + run_num) * input_batch_size] = image_features[a]
        final_text_embeddings[(1 + run_num) * input_batch_size : (2 + run_num) * input_batch_size] = text_embedding[b]
        output_attention_mask[(1 + run_num) * input_batch_size : (2 + run_num) * input_batch_size] = \
            input_attention_mask[b]

    return final_image_features, final_text_embeddings, output_attention_mask, ground_truths


def compute_ranks(sims):
    # assert imgs.shape == rcps.shape, 'recipe features and image features should have same dimension'
    # # pdb.set_trace()
    # imgs = imgs / np.linalg.norm(imgs, axis=1)[:, None]
    # rcps = rcps / np.linalg.norm(rcps, axis=1)[:, None]
    # if retrieved_type == 'recipe':
    #     sims = np.dot(imgs, rcps.T)  # [N, N]
    # else:
    #     sims = np.dot(rcps, imgs.T)

    ranks = []
    preds = []
    # loop through the N similarities for images
    for ii in range(sims.shape[0]):
        # get a column of similarities for image ii
        sim = sims[ii, :]
        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()
        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)
        ranks.append(pos + 1.0)
        preds.append(sorting[0])
    # pdb.set_trace()
    return np.asarray(ranks), preds


def rank(rcps, imgs, model=None, retrieved_type='recipe', retrieved_range=1000, verbose=False):
    t1 = time()
    N = retrieved_range
    data_size = imgs.shape[0]
    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    # pickler(imgs, 'image_outputs.pkl')
    # pickler(rcps, 'recipe_outputs.pkl')
    # if draw_hist:
    #     plt.figure(figsize=(16, 6))
    # average over 10 sets
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        probs = np.zeros((N, N))
        for x in range(N):
            if retrieved_type == 'recipe':
                probs[x] = model(imgs_sub[x].repeat(N, 1, 1), rcps_sub)[:, 1]
            else:
                probs[x] = model(imgs_sub, rcps_sub[x].repeat(N, 1, 1))[:, 1]
        # loop through the N similarities for images
        ranks, _ = compute_ranks(probs)

        # pickler(ranks, 'ranks.pkl')
        # pickler(ids_sub, 'indices.pkl')

        recall = {1: 0.0, 5: 0.0, 10: 0.0}
        for ii in recall.keys():
            recall[ii] = (ranks <= ii).sum() / ranks.shape[0]
        med = int(np.median(ranks))
        for ii in recall.keys():
            glob_recall[ii] += recall[ii]
        glob_rank.append(med)
        # if draw_hist:
        #     ranks = np.array(ranks)
        #     plt.subplot(2, 5, i + 1)
        #     n, bins, patches = plt.hist(x=ranks, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        #     plt.grid(axis='y', alpha=0.75)
        #     plt.ylim(top=300)
        #     plt.text(23, 45, 'avgR(std) = {:.2f}({:.2f})\nmedR={:.2f}\n#<{:d}:{:d}|#={:d}:{:d}|#>{:d}:{:d}'.format(
        #         np.mean(ranks), np.std(ranks), np.median(ranks),
        #         med, (ranks < med).sum(), med, (ranks == med).sum(), med, (ranks > med).sum()))
    # if draw_hist:
    #     plt.savefig(f'hist_{epoch}.png')

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10

    medR = np.mean(glob_rank)
    medR_std = np.std(glob_rank)
    t2 = time()
    if verbose:
        print(f'=>retrieved_range={retrieved_range}, MedR={medR:.4f}({medR_std:.4f}), time={t2 - t1:.4f}s')
    return medR, medR_std, glob_recall