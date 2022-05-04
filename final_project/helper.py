import torch
import random
import numpy as np
from time import time
from torch import nn
from tqdm import tqdm


def get_transformer_input(image_features, text_embedding, input_attention_mask):
    num_negative_to_positive_sample_ratio = 1

    input_batch_size = image_features.shape[0]
    output_batch_size = (num_negative_to_positive_sample_ratio + 1) * input_batch_size
    ground_truths = torch.zeros(output_batch_size)
    ground_truths[:input_batch_size] = 1

    final_image_features = torch.zeros(output_batch_size, *image_features.shape[1:])
    final_text_embeddings = torch.zeros(output_batch_size, *text_embedding.shape[1:])
    output_attention_mask = torch.zeros(output_batch_size, *input_attention_mask.shape[1:])

    final_image_features[:input_batch_size] = image_features
    final_text_embeddings[:input_batch_size] = text_embedding
    output_attention_mask[:input_batch_size] = input_attention_mask

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


def save_model(model, fpath):
    torch.save(model, fpath)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


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


def rank(rcps: list, imgs: list, attention_masks: list, model=None, retrieved_type='recipe', retrieved_range=100,
         verbose=False, device='cuda'):

    # save_model({'rcps': rcps, 'imgs': imgs, 'attention_masks': attention_masks}, 'data.pt')
    # rcps = torch.cat(rcps, dim=0)
    # imgs = torch.cat(imgs, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    t1 = time()
    N = retrieved_range
    data_size = len(imgs)
    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    softmax = nn.Softmax(dim=-1)
    # pickler(imgs, 'image_outputs.pkl')
    # pickler(rcps, 'recipe_outputs.pkl')
    # if draw_hist:
    #     plt.figure(figsize=(16, 6))
    # average over 10 sets
    for i in range(2):
        ids_sub = np.random.choice(data_size, N, replace=False)
        # imgs_sub = imgs[ids_sub, :]
        # rcps_sub = rcps[ids_sub, :]
        imgs_sub = [imgs[ind] for ind in ids_sub]
        rcps_sub = [rcps[ind] for ind in ids_sub]
        attention_masks_sub = [attention_masks[ind] for ind in ids_sub]
        probs = np.zeros((N, N))
        for x in tqdm(range(N)):
            for y in range(N):
                # if retrieved_type == 'recipe':
                #     probs[x] = model(imgs_sub[x].repeat(N, 1, 1), rcps_sub)[:, 1]
                # else:
                #     probs[x] = model(imgs_sub, rcps_sub[x].repeat(N, 1, 1))[:, 1]
                try:
                    if retrieved_type == 'recipe':
                        probs[x][y] = softmax(model(imgs_sub[x].unsqueeze(0).to(device), rcps_sub[y].unsqueeze(0).to(device),
                                                    ~attention_masks_sub[y].bool().unsqueeze(0).to(device)))[0, 1]
                    else:
                        probs[x][y] = softmax(model(imgs_sub[y].unsqueeze(0).to(device), rcps_sub[x].unsqueeze(0).to(device),
                                                    ~attention_masks_sub[y].bool().unsqueeze(0).to(device)))[0, 1]
                except RuntimeError as e:
                    print(imgs_sub[x].unsqueeze(0).shape, rcps_sub[y].unsqueeze(0).shape, attention_masks_sub[y].unsqueeze(0).shape)
                    print(attention_masks_sub)
                    print(ids_sub, x, y)
                    raise(RuntimeError(str(e)))
        # loop through the N similarities for images
        ranks, _ = compute_ranks(probs)

        recall = {1: 0.0, 5: 0.0, 10: 0.0}
        for ii in recall.keys():
            recall[ii] = (ranks <= ii).sum() / ranks.shape[0]
        med = int(np.median(ranks))
        for ii in recall.keys():
            glob_recall[ii] += recall[ii]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10

    medR = np.mean(glob_rank)
    medR_std = np.std(glob_rank)
    t2 = time()
    if verbose:
        print(f'=>retrieved_range={retrieved_range}, MedR={medR:.4f}({medR_std:.4f}), time={t2 - t1:.4f}s')
        print(f'Global recall: 1: {glob_recall[1]:.4f}, 5: {glob_recall[5]:.4f}, 10: {glob_recall[10]:.4f}')
    return medR, medR_std, glob_recall

def calculate_metrics(image_encoder, text_encoder, cm_transformer, dataloader, tokenizer, device='cuda'):
    print('Calculating Metrics')
    image_encoder.eval()
    text_encoder.eval()
    cm_transformer.eval()

    text_embeddings = list()
    image_features = list()
    attention_masks = list()

    with torch.no_grad():
        for text, image in tqdm(dataloader):
            text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            text_outputs = text_encoder(**text_inputs)
            image_outputs = image_encoder(image.to(device))

            for text_output, image_feature, attention_mask in zip(text_outputs, image_outputs, text_inputs.attention_mask):
                text_embeddings.append(text_output.cpu())
                image_features.append(image_feature.cpu())
                attention_masks.append(attention_mask.cpu())

        return rank(text_embeddings, image_features, attention_masks, model=cm_transformer, device=device, verbose=True)
        
