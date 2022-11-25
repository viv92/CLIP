### Program implementing CLIP (Contrastive Language Image Pretraining) - using ResNet as image encoder

## Features:
# 1. Multi-class N pair loss / contrastive loss over minibatch of image-text embedding pairs.

## Todos / Questions:
# 1. Image encoder: Vision Transformer versus ResNet
# 2. Text encoder: Transformer encoder with subsequent mask (masked self-attention for promoting language modelling skills in the encoder)
# 2.1 So do we have an auxiliary loss for language modelling along with the clip loss ? (paper does not use an explicit auxiliary loss, but the masked self-attention is supposed to implicitly promote language modelling abilities in transformer encoder)
# 2.2 Swapping the order of layernorm in sublayer connection to apply layernorm first (this would require a final layer norm in the encoder out)
# 2.3. Text encoding = encoder output for the [EOS] index (not using the entire ouput vector of shape [max_seq_len, vocab_size])
# 2.4. Another version - use BERT encoder: Bert style masking instead of subsequent mask in transformer encoder
# 3. add layernorm and dropout to image embeddings (as done for text embeddings)
# 4. logits are pairwise cosine similarities, scalled by exp(temperature). Temperature is a learnt parameter, not a tuned hyperparameter.
# 5. Contrastive loss versus CrossEntropyLoss - similarities and differences: refer register note.
# 6. GELU non-linearity in the feedforward layers of transformer
# 7. Replace avg pooling in resnet with attention pooling
# 8. Optimize code - faster vector normalization
# 9. cross check with moen's code.
# 10. img preprocessing: does it make a difference if img.shape: [c,h,w] versus [c,w,h] - What does resnet / ViT expect ?
# 11. Why test_trials > 1 give CUDA_OUT_OF_MEMORY error ?
# 12. Learning schedule versus saving and loading optimizer state - possible conflicts


import os
import math
import numpy as np
import cv2
import json
import unidecode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import sentencepiece as spm

# for learning rate scheduling
from torch.optim.lr_scheduler import LambdaLR

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len]
    mask = mask.unsqueeze(0).expand(batch_size, max_seq_len, max_seq_len) # mask.shape = [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token).unsqueeze(1) # mask.shape: [batch_size, 1, max_seq_len]
    mask = mask.expand(batch_size, max_seq_len, max_seq_len) # mask.shape: [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# forward hook for reading resnet penultimate layer logits
def forward_hook(module, input, output):
    global resnet_avgpool_output
    resnet_avgpool_output = output

# class for image embeddings
class ImageEmbeddings(nn.Module):
    def __init__(self, hook_fn, d_model):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.avgpool.register_forward_hook(hook_fn)
        self.proj = nn.Linear(2048, d_model, bias=False)
    def forward(self, imgs): # imgs.shape: [b,c,w,h]
        _ = self.resnet(imgs)
        emb = resnet_avgpool_output # emb.shape: [b, 2048, 1, 1]
        emb = emb.flatten(start_dim=1, end_dim=-1) # emb.shape: [b, 2048]
        emb = self.proj(emb) # emb.shape: [b, d_model]
        return emb

# class for caption embeddings
class CaptionEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, dropout, device):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.device = device
    def forward(self, x):
        batch_size, max_seq_len = x.shape[0], x.shape[1]
        tok_emb = self.tok_emb(x) # tok_emb.shape: [batch_size, max_seq_len, d_model]
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        final_emb = self.dropout( self.norm(tok_emb + pos_emb) )
        final_emb = final_emb * math.sqrt(self.d_model)
        return final_emb


# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))

# class implementing multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.output = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.attn_weights = None # placeholder to store attention weights (used to visualize attention matrices)
        self.dropout = nn.Dropout(dropout)

    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        # key.shape: [batch_size, seq_len, d_model]; mask.shape: [batch_size, seq_len, seq_len]
        # project key, query, value and reshape into multiple heads
        batch_size = key.shape[0]
        # (batch_size, seq_len, d_model) -proj-> (batch_size, seq_len, proj_dim) -view-> (batch_size, seq_len, n_heads, d_k) -transpose-> (batch_size, n_heads, seq_len, d_k)
        proj_key = self.W_K(key).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_query = self.W_Q(query).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_value = self.W_V(value).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2)
        # expand mask for n_heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # mask.shape: [batch_size, n_heads, seq_len, seq_len]
        # calculate attention
        attn_multihead, attn_weights = self.scaled_dotprod_attn(proj_key, proj_query, proj_value, mask, self.dropout)
        attn_multihead = attn_multihead.transpose(1, 2) # attn_multihead.shape: [batch_size, seq_len, n_heads, d_v]
        attn_multihead = torch.flatten(attn_multihead, start_dim=-2, end_dim=-1) # attn_multihead.shape: [batch_size, seq_len, n_heads * d_v]
        attn_multihead = self.output(attn_multihead) # attn_multihead.shape: [batch_size, seq_len, d_model]
        self.attn_weights = attn_weights
        return attn_multihead

    # function to calculate scaled dot product attention for one head
    def scaled_dotprod_attn(self, key, query, value, mask=None, dropout=None): # key.shape: [batch_size, n_heads, seq_len, d_k]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) # attn_scores.shape: [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_weights = attn_scores.softmax(dim=-1) # attn_weights.shape: [batch_size, n_heads, seq_len, seq_len]
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        attn_vector = torch.matmul(attn_weights, value) # attn_vector.shape: [batch_size, n_heads, seq_len, d_v]
        return attn_vector, attn_weights

# class implementing Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * ( (x - mean) / (std + self.eps) ) + self.b

# class implementing residual + normalization connection - takes in any block and applies a residual connection around it + a layer normalization on top
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return self.norm( x + self.dropout(sublayer(x)) )

# class implementing a single encoder layer
# each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask)) # x.shape: [batch_size, seq_len, d_model]
        x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
        return x

# class implementing the entire encoder block = stacked encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# class implemeting the entire model
class CLIP(nn.Module):
    def __init__(self, text_encoder, image_embeddings_model, caption_embeddings_model, d_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_embeddings = image_embeddings_model
        self.caption_embeddings = caption_embeddings_model
        self.eos_proj = nn.Linear(d_model, d_model, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.))
    # def inverse_projection(self, decoder_out):
    #     W = self.caption_embeddings.tok_emb.weight
    #     scores = torch.matmul(decoder_out, W.T)
    #     return scores
    def encode(self, src, src_mask):
        encoder_out = self.text_encoder(src, src_mask)
        return encoder_out
    def forward(self, encoded_img, cap_embs, cap_mask):
        encoder_out = self.encode(cap_embs, cap_mask) # encoder_out.shape: [batch_size, seq_len, d_model]
        encoded_txt = self.eos_proj(encoder_out[:, 0]) # encoded_txt.shape: [batch_size, d_model]
        # cosine similarity score matrix
        encoded_img_norm = torch.linalg.vector_norm(encoded_img, dim=1).unsqueeze(1) # shape: [batch_size, 1]
        encoded_txt_norm = torch.linalg.vector_norm(encoded_txt, dim=1).unsqueeze(1)
        norm_matrix = torch.matmul(encoded_img_norm, encoded_txt_norm.T)
        cosine_similarity = torch.matmul(encoded_img, encoded_txt.T) / norm_matrix
        scores = cosine_similarity * torch.exp(self.temperature)
        return scores
    # function used for test time label prediction (1 = correct pair, 0 = wrong pair)
    def predict(self, encoded_img, cap_embs, cap_mask, threshold=0.9):
        scores = self.forward(encoded_img, cap_embs, cap_mask)
        scores = scores.trace()
        preds = torch.gt(scores, threshold).int()
        return preds

# caller function to instantiate the CLIP model, using the defined hyperparams as input
def init_CLIP(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    image_embeddings_model = ImageEmbeddings(forward_hook, d_model)
    caption_embeddings_model = CaptionEmbeddings(vocab_size, max_seq_len, d_model, dropout, device)
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers) # encoder = stacked encoder layers
    model = CLIP(encoder, image_embeddings_model, caption_embeddings_model, d_model) # the CLIP model
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# function to obtain a batch of img embeddings from a batch of img filenames
def get_img_embeddings(img_filenames, model, max_seq_len, pad_token, device):
    imgs_folder = 'dataset_coco_val2017/images/'
    # load images from image filenames
    imgs = []
    for img_file in img_filenames:
        img_path = imgs_folder + img_file
        img = cv2.imread(img_path, 1)
        img_size = 224
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.transpose(1,2).transpose(0,1).transpose(1,2) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
        ])
        img = transforms(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0).to(device) # [b,c,h,w]
    img_embs = model.image_embeddings(imgs) # [b, d_model]
    return img_embs

# function to convert caption text to tokens
def tokenize_captions():
    pass

# utility function to load img and captions data
def load_data():
    imgs_folder = 'dataset_coco_val2017/images/'
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)

    img_dict, img_cap_dict = {}, {}
    max_caption_len = 0

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name

    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # update max_caption_len
        caption_len = len(caption.split(' '))
        if caption_len > max_caption_len:
            max_caption_len = caption_len
        # process caption
        caption = unidecode.unidecode(caption) # strip accents
        caption = caption.lower()
        # use img_name as key for img_cap_dict
        img = img_dict[id]
        img_cap_dict[img] = caption

    max_caption_len += 3 # for <s>, </s> and a precautionary <pad>
    return img_cap_dict, max_caption_len


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device):
    augmented_imgs, tokenized_captions = [], []
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename, caption_text in minibatch:
        # tokenize caption text
        caption_tokens = spm_processor.encode(caption_text, out_type=int)
        caption_tokens = [sos_token] + caption_tokens + [eos_token] # append sos and eos tokens
        while len(caption_tokens) < max_seq_len:
            caption_tokens.append(pad_token) # padding
        tokenized_captions.append(caption_tokens)
        # obtain augmented img from img_filename
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.transpose(1,2).transpose(0,1).transpose(1,2) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
        ])
        img = transforms(img)
        augmented_imgs.append(img)
    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    tokenized_captions = torch.LongTensor(tokenized_captions).to(device)
    # get img embeddings
    img_embs = model.image_embeddings(augmented_imgs) # [b, d_model] # img_embs.shape: [batch_size, d_model]
    # get caption mask
    cap_pad_mask = pad_mask(tokenized_captions, pad_token) # pad mask for captions
    cap_sub_mask = subsequent_mask(tokenized_captions.shape).to(device)
    cap_mask = torch.logical_and( cap_pad_mask, cap_sub_mask ) # add subsequent mask for captions
    # get caption embeddings
    cap_embs = model.caption_embeddings(tokenized_captions) # shape: [batch_size, max_seq_len, d_model]
    return img_embs, cap_embs, cap_mask


# # utility function to process batch of raw data and yield batch of data embeddings along with masks
# def process_batch(model, img_filenames, captions, batch_size, max_seq_len, d_model, pad_token, device):
#     # get img embeddings
#     img_embs = get_img_embeddings(img_filenames, model, max_seq_len, pad_token, device) # img_embs.shape: [batch_size, d_model]
#     # get caption mask
#     cap_pad_mask = pad_mask(captions, pad_token) # pad mask for captions
#     cap_sub_mask = subsequent_mask(captions.shape).to(device)
#     cap_mask = torch.logical_and( cap_pad_mask, cap_sub_mask ) # add subsequent mask for captions
#     # get caption embeddings
#     cap_embs = model.caption_embeddings(captions) # shape: [batch_size, max_seq_len, d_model]
#     return img_embs, cap_embs, cap_mask


# function to calculate loss by forward propping through the model
def calculate_loss(model, img_embs, cap_embs, cap_mask, batch_size, device):
    # feed embeddings to transformer decoder
    scores = model(img_embs, cap_embs, cap_mask) # scores.shape: [batch_size, batch_size]
    # targets
    targets = torch.arange(batch_size).long().to(device)
    # cross entropy loss
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_img = criterion(scores, targets)
    loss_txt = criterion(scores.T, targets)
    loss = (loss_img + loss_txt) / 2
    return loss

# utility function for learning rate schedule - from attention paper
def lr_rate(curr_step, warmup_steps, lr_init, d_model):
    curr_step += 1 # one-indexed instead of zero indexed
    lr_new = lr_init * ( d_model**(-0.5) * min( curr_step**(-0.5), curr_step * (warmup_steps**(-1.5)) ) )
    return lr_new

# function to calculate test accuracy
def calculate_test_accuracy(test_trials, model, test_data):
    test_accuracy = []
    for ep in range(test_trials):
        # fetch minibatch
        idx = np.arange(len(test_data))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [test_data[i] for i in idx]
        gt_labels = np.ones(batch_size)
        # flip labels at random
        for i in range(batch_size):
            r = np.random.choice(np.arange(2))
            if r == 0:
                # flip label
                gt_labels[i] = 0
                # flip caption
                flipped_idx = int((i+1) % batch_size)
                minibatch[i][1] = minibatch[flipped_idx][1]
        # get predicted labels
        threshold = 0.9
        # process batch to get batch of embeddings and masks
        img_embs, cap_embs, cap_mask = process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device)
        # feed to transformer decoder
        pred_labels = model.predict(img_embs, cap_embs, cap_mask, threshold) # pred_labels.shape: [batch_size]
        # accuracy
        gt_labels = torch.from_numpy(gt_labels).int().to(device)
        accuracy = torch.eq(gt_labels, pred_labels.detach()).int().sum()
        accuracy = (accuracy.item() * 100) / batch_size
        test_accuracy.append(accuracy)
    mean_accuracy = sum(test_accuracy) / len(test_accuracy)
    return mean_accuracy

# utility function to save a checkpoint (model_state and optimizer_state) - saves on whatever device the model was training on
def save_checkpoint(checkpoint_path, model, optimizer):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, checkpoint_path)

# utility function to load checkpoint to resume training from - loads to the device passed as 'device' argument
def load_checkpoint(checkpoint_path, model, optimizer, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    model.train()
    return model, optimizer


### main ###
if __name__ == '__main__':

    # hperparams
    img_size = 224 # resize for resnet
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 16
    train_split = .95
    test_trials = 10
    num_epochs = 24000 * 7
    random_seed = 10
    lr_init = 1e-3 # not using scheduled lr since possible conflict on saving / loading model
    warmup_steps = num_epochs // 3

    checkpoint_path = 'ckpts_clip_resnet/latest.pt' # path to a saved checkpoint (model state and optimizer state) to resume training from
    resume_training_from_ckpt = True

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # spm processor as tokenizer / detokenizer
    spm_processor = spm.SentencePieceProcessor(model_file='spm1.model')
    vocab_size = len(spm_processor)

    # load data and create img_cap_dict
    img_cap_dict, max_seq_len = load_data()

    # tokenize captions and append sos, eos, pad tokens
    sos_token, eos_token, unk_token = spm_processor.piece_to_id(['<s>', '</s>', '<unk>'])
    pad_token = unk_token # <unk> token is used as the <pad> token

    # train-test split
    train_data, test_data = [], []
    n_total_keys = len(img_cap_dict.keys())
    n_train_keys = int(n_total_keys * train_split)
    for i, (k, v) in enumerate(img_cap_dict.items()):
        if i <= n_train_keys:
            train_data.append([k, v])
        else:
            test_data.append([k, v])

    # free memory occupied by img_cap_dict as its no longer needed
    del img_cap_dict

    # instantiate model
    model = init_CLIP(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # instantiate optimizer and lr_scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_init, betas=(.9, .98), eps=1e-9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_init, betas=(.9, .98), eps=1e-9, weight_decay=1e-4)
    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lr_rate(x, warmup_steps, lr_init, d_model) ) # x increases like a counter on each call to lr_scheduler.step()

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        model, optimizer = load_checkpoint(checkpoint_path, model, optimizer, device)

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(n_train_keys)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [train_data[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - and obtain their embeddings
        img_embs, cap_embs, cap_mask = process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device) # imgs.shape:[batch_size, 3, 64, 64], captions.shape:[batch_size, max_seq_len]

        # calculate loss
        loss = calculate_loss(model, img_embs, cap_embs, cap_mask, batch_size, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()


        if ep % (num_epochs//20) == 0:

            # print intermediate loss
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

            # save checkpoint
            save_checkpoint(checkpoint_path, model, optimizer)

            # test - get test accuracy
            model.eval()
            accuracy = calculate_test_accuracy(test_trials, model, test_data)
            print('accuracy: ', accuracy)
            model.train()
