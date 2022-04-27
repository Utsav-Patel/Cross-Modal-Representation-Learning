import torch
import random


def get_transformer_input(image_features, text_embedding):
    num_negative_to_positive_sample_ratio = 2

    input_batch_size = image_features.shape[0]
    output_batch_size = (num_negative_to_positive_sample_ratio + 1) * input_batch_size
    ground_truths = torch.zeros(output_batch_size)
    ground_truths[:input_batch_size] = 1

    final_image_features = torch.zeros(output_batch_size, *image_features.shape[1:])
    final_text_embeddings = torch.zeros(output_batch_size, *text_embedding.shape[1:])

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

    return final_image_features, final_text_embeddings, ground_truths

