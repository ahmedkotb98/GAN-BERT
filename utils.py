import io
import json
import random

import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs


target_names = ["Neutral", "Positive", "Extremely Negative", "Negative", "Extremely Positive"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_prediction(model, tokenizer, input, discriminator):
    encoded_dict = tokenizer.encode_plus(
                    input,                     
                    add_special_tokens = True, 
                    max_length = 64,           
                    pad_to_max_length = True,
                    return_attention_mask = True,   
                    return_tensors = 'pt',     
                )

    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoded_dict.items()}
    del inputs_onnx['token_type_ids']
    ort_outs = model.run(['output'], inputs_onnx)
    hidden_states = torch.from_numpy(ort_outs[-1])
    _, logits, probs = discriminator(hidden_states)
    filtered_logits = logits[:,0:-1]
    _, preds = torch.max(filtered_logits, 1)

    return target_names[preds.argmax()]