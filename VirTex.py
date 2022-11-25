### Program implementing VirTex - Learning Visual Representations from Textual Annotations

## Features:
# 1. Train a visual backbone and a textual head from SCRATCH for image captioning - the learnt visual features transfer better to downstream tasks compared to visual features learnt from imagenet
# 2. Visual backbone is ResNet-50 and textual head is a pair of transformer decoders that operate in a bi-directional manner
# 3. The outputs of the two transformer decoders are not concatenated. Each transformer has a separate loss function, but the training gradient passes through a common visual backbone.
# 4. A projection layer is used to project visual features to the same dimension as the textual embeddings - this projection layer is discarded in downstream tasks
# 7. Vocabulary is based on sentencepiece tokens obtained using BPE algorithm
# 8. Dataset: Coco Captions train2017 + data augmentation
# 10. Vocabulary built using SPM has internally created tokens for <s>, </s> and <unk>. We use the start <s> token, the end </s> token and <unk> = pad token

## Todos / Questions:
# 1. does multiplying embeddings with math.sqrt(d_model) create mismatch when inverting the output to embeddings in the decoder head? [Probably not]
# 2. [done] first complete the training pipeline for forward decoder. Then we'll add in the pipeline for backward decoder.
# 5. Use GELU activations in transformer decoders instead of ReLU
# 6. [done] Textual embeddings = dropout( LayerNorm( learn_token_embedding + learnt positional_embedding ) ).
# 7. [done] The same token embedding matrix is used at the input and the output of both of the transformer decoders.
# 9. Optimization: SGD + momentum + weight decay lookahead + learning rate warmup + distributed training
# 10. [done - using ignore_index in CrossEntropyLoss] loss should not include loss over pad tokens
# 11. Label smoothing
# 12. Temperature annealing in softmax(scores) during training to encourage exploration

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
    global resnet_layer4_output
    resnet_layer4_output = output

# class for image embeddings
class ImageEmbeddings(nn.Module):
    def __init__(self, hook_fn, d_model):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.layer4.register_forward_hook(hook_fn)
        self.proj = nn.Linear(2048, d_model, bias=False)
    def forward_train(self, imgs): # imgs.shape: [b,c,w,h]
        _ = self.resnet(imgs)
        emb = resnet_layer4_output
        emb = emb.transpose(1,2).transpose(2,3)
        emb = self.proj(emb)
        return emb
    def forward_downstream(self, imgs): # imgs.shape: [b,c,w,h]
        _ = self.resnet(imgs)
        emb = resnet_layer4_output
        emb = emb.transpose(1,2).transpose(2,3)
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
    def scaled_dotprod_attn(self, key, query, value, mask=None, dropout=None): # key.shape: [batch_size, seq_len, proj_dim]
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

# class implementing a single decoder layer
# each decoder layer has three blocks: 1. (self) (masked) multihead attention 2. (src) (unmasked) multihead attention  3. feed_forward; with sublayer connection around each
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 3) # one for self_attn block, second for src_attn block, third for feed_forward block
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        m = encoder_out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # first apply self_attn block
        x = self.sublayers[1](x, lambda x: self.src_attn(m, x, m, src_mask)) # src_attn: (key from encoder, query from decoder, value from encoder)
        x = self.sublayers[2](x, self.feed_forward)
        return x

# class implementing the entire decoder block = stacked decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return x

# class implemeting the entire model
class VirTex(nn.Module):
    def __init__(self, decoder_fwd, decoder_rev, image_embeddings_model, caption_embeddings_model, tgt_vocab_size):
        super().__init__()
        self.decoder_fwd = decoder_fwd
        self.decoder_rev = decoder_rev
        self.image_embeddings = image_embeddings_model
        self.caption_embeddings = caption_embeddings_model
        # self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def inverse_projection(self, decoder_out):
        W = self.caption_embeddings.tok_emb.weight
        scores = torch.matmul(decoder_out, W.T)
        return scores
    def decode(self, tgt_fwd, tgt_rev, encoder_out, src_mask, tgt_mask):
        decoder_out_fwd = self.decoder_fwd(tgt_fwd, encoder_out, src_mask, tgt_mask)
        decoder_out_rev = self.decoder_rev(tgt_rev, encoder_out, src_mask, tgt_mask)
        return decoder_out_fwd, decoder_out_rev
    def forward(self, src, src_mask, tgt_fwd, tgt_rev, tgt_mask):
        decoder_out_fwd, decoder_out_rev = self.decode(tgt_fwd, tgt_rev, src, src_mask, tgt_mask) # decoder_out.shape: [batch_size, seq_len, d_model]
        scores_fwd = self.inverse_projection(decoder_out_fwd) # scores.shape: [batch_size, seq_len, tgt_vocab]
        scores_rev = self.inverse_projection(decoder_out_rev)
        return scores_fwd, scores_rev
    # function used for test time generation - outputs score only for last element in seq
    def generate(self, src, src_mask, tgt, tgt_mask, pred_index):
        decoder_out = self.decoder_fwd(tgt, src, src_mask, tgt_mask) # decoder_out.shape: [batch_size, seq_len, d_model]
        score = self.inverse_projection(decoder_out[:, pred_index]) # score.shape: [batch_size, d_model]
        return score

# caller function to instantiate the transformer model, using the defined hyperparams as input
def make_model(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    image_embeddings_model = ImageEmbeddings(forward_hook, d_model)
    caption_embeddings_model = CaptionEmbeddings(vocab_size, max_seq_len, d_model, dropout, device)
    decoder_layer = DecoderLayer(deepcopy(attn), deepcopy(attn), deepcopy(ff), d_model, dropout) # single decoder layer
    decoder_fwd = Decoder(decoder_layer, n_layers) # decoder = stacked decoder layers
    decoder_rev = Decoder(decoder_layer, n_layers) # decoder = stacked decoder layers
    model = VirTex(decoder_fwd, decoder_rev, image_embeddings_model, caption_embeddings_model, vocab_size) # the transformer decoder model
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
        resize_shape = (224, 224)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.transpose(1,2).transpose(0,1) # [w,h,c] -> [c,w,h]
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0).to(device) # [b,c,w,h]
    img_embs = model.image_embeddings.forward_train(imgs) # [b, 7, 7, d_model]
    img_embs = img_embs.flatten(start_dim=1, end_dim=2) # [b, 7*7, d_model]
    return img_embs

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


# utility function to process batch of raw data and yield batch of data embeddings along with masks
def process_batch(model, img_filenames, captions_fwd, captions_rev, batch_size, max_seq_len, d_model, pad_token, device):
    # get img embeddings
    img_embs = get_img_embeddings(img_filenames, model, max_seq_len, pad_token, device) # img_embs.shape: [batch_size, 7*7, d_model]
    # pad img embeddings
    assert(max_seq_len > img_embs.shape[1])
    pad_dim = max_seq_len - img_embs.shape[1]
    pad_tensor = torch.ones(batch_size, pad_dim, d_model) * pad_token
    pad_tensor = pad_tensor.to(device)
    img_embs = torch.cat((img_embs, pad_tensor), dim=1) # img_embs.shape: [batch_size, max_seq_len, d_model]
    # get img mask
    img_mask = pad_mask(img_embs[:,:,0], pad_token) # img_mask.shape: [batch_size, max_seq_len, max_seq_len]
    # get caption mask
    cap_pad_mask = pad_mask(captions_fwd, pad_token) # pad mask for captions
    cap_sub_mask = subsequent_mask(captions_fwd.shape).to(device)
    cap_mask = torch.logical_and( cap_pad_mask, cap_sub_mask ) # add subsequent mask for captions
    # get caption embeddings
    cap_embs_fwd = model.caption_embeddings(captions_fwd) # shape: [batch_size, max_seq_len, d_model]
    cap_embs_rev = model.caption_embeddings(captions_rev)
    return img_embs, img_mask, cap_embs_fwd, cap_embs_rev, cap_mask


# function that does everything required to calculate the loss
def calculate_loss(model, minibatch, batch_size, max_seq_len, d_model, pad_token, device):
    # unpack
    img_filenames, captions = map(list, zip(*minibatch)) # img_filenames.shape: [batch_size]; captions.shape: [batch_size, 2, max_seq_len]
    captions_fwd, captions_rev = map(torch.LongTensor, zip(*captions))
    captions_fwd, captions_rev = captions_fwd.to(device), captions_rev.to(device)
    # process batch to get batch of embeddings and masks
    img_embs, img_mask, cap_embs_fwd, cap_embs_rev, cap_mask = process_batch(model, img_filenames, captions_fwd, captions_rev, batch_size, max_seq_len, d_model, pad_token, device)
    # feed to transformer decoder
    scores_fwd, scores_rev = model(img_embs, img_mask, cap_embs_fwd, cap_embs_rev, cap_mask) # scores.shape: [batch_size, max_seq_len, vocab_size]
    scores_fwd = scores_fwd.transpose(1, 2) # scores.shape: [batch_size, vocab_size, max_seq_len]
    scores_rev = scores_rev.transpose(1, 2)
    # target: right shifted captions
    pad_dim = 1
    pad_tensor = torch.ones(batch_size, pad_dim, dtype=torch.long) * pad_token
    pad_tensor = pad_tensor.to(device)
    target_fwd = torch.cat( (captions_fwd[:, 1:], pad_tensor), dim=1 ) # target.shape: [batch_size, max_seq_len]
    target_rev = torch.cat( (captions_rev[:, 1:], pad_tensor), dim=1 )
    # preview a prediction versus target
    score = scores_fwd[0]
    logprobs = F.log_softmax(score, dim=0)
    _, preview_pred_tokens = torch.max(logprobs, dim=0) # pred_tokens.shape: [max_seq_len]
    preview_tgt_tokens = target_fwd[0]
    # cross entropy loss
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_token) # doesn't count loss for pad_token
    loss_fwd = criterion(scores_fwd, target_fwd)
    loss_rev = criterion(scores_fwd, target_rev)
    loss = loss_fwd + loss_rev
    return loss, preview_pred_tokens, preview_tgt_tokens

# utility function for learning rate schedule - from attention paper
def lr_rate(curr_step, warmup_steps, lr_init, d_model):
    curr_step += 1 # one-indexed instead of zero indexed
    lr_new = lr_init * ( d_model**(-0.5) * min( curr_step**(-0.5), curr_step * (warmup_steps**(-1.5)) ) )
    return lr_new

### main ###
if __name__ == '__main__':

    # hperparams
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    train_split = .95
    test_trials = 5
    num_epochs = 500
    random_seed = 10
    lr_init = 1. # scheduled
    warmup_steps = num_epochs // 3

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

    for img, caption in img_cap_dict.items():
        caption_tokens = spm_processor.encode(caption, out_type=int)
        caption_tokens_fwd = [sos_token] + caption_tokens + [eos_token]
        # rev_caption_tokens = caption_tokens[::-1]
        caption_tokens_rev = [eos_token] + caption_tokens[::-1] + [sos_token]
        while len(caption_tokens_fwd) < max_seq_len:
            caption_tokens_fwd.append(pad_token)
            caption_tokens_rev.append(pad_token)
        img_cap_dict[img] = [caption_tokens_fwd, caption_tokens_rev] # each value in the dict is a pair of fwd and rev captions

    # train-test split
    train_data, test_data = [], []
    n_total_keys = len(img_cap_dict.keys())
    n_train_keys = int(n_total_keys * train_split)
    for i, (k, v) in enumerate(img_cap_dict.items()):
        if i <= n_train_keys:
            train_data.append([k, v])
        else:
            test_data.append([k, v])

    # instantiate model
    model = make_model(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # instantiate optimizer and lr_scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_init, betas=(.9, .98), eps=1e-9, weight_decay=1e-4)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lr_rate(x, warmup_steps, lr_init, d_model) ) # x increases like a counter on each call to lr_scheduler.step()

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(n_train_keys)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [train_data[i] for i in idx]

        # calculate loss
        loss, preview_pred_tokens, preview_tgt_tokens = calculate_loss(model, minibatch, batch_size, max_seq_len, d_model, pad_token, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print intermediate loss
        if ep % 50 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))
            print('\npreview_pred_tokens: ', preview_pred_tokens)
            print('preview_tgt_tokens: ', preview_tgt_tokens)

    # test
    for i in range(test_trials):

        sample_id = np.random.choice(len(test_data))
        img_file, gt_caption = test_data[sample_id]
        gt_caption = gt_caption[0] # just using the fwd caption for testing

        imgs_folder = 'dataset_coco_val2017/images/'
        img_path = imgs_folder + img_file
        img = cv2.imread(img_path, 1)
        resize_shape = (224, 224)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255

        img_file = [img_file]
        pred_caption_fwd = [pad_token for i in range(max_seq_len)]
        pred_caption_rev = [pad_token for i in range(max_seq_len)]
        pred_caption_fwd[0], pred_caption_rev[0] = sos_token, eos_token # init with start token (or end token for reversed caption)
        pred_caption_fwd = torch.LongTensor(pred_caption_fwd).unsqueeze(0).to(device)
        pred_caption_rev = torch.LongTensor(pred_caption_rev).unsqueeze(0).to(device)

        for pred_index in range(max_seq_len-1):
            img_embs, img_mask, cap_embs, cap_embs_rev, cap_mask = process_batch(model, img_file, pred_caption_fwd, pred_caption_rev, 1, max_seq_len, d_model, pad_token, device)
            score = model.generate(img_embs, img_mask, cap_embs, cap_mask, pred_index) # score.shape: [1, vocab_size]
            logprobs = F.log_softmax(score, dim=-1)
            _, next_word_token = torch.max(logprobs, dim=-1) # greedy decoding for now
            # update pred_caption
            pred_caption_fwd[0][pred_index+1] = next_word_token

        # decode pred_caption from tokens to text
        pred_caption_fwd = list(pred_caption_fwd.squeeze(0).detach().cpu().numpy())
        # pred_caption_text = spm_processor.decode(pred_caption)
        gt_caption_text = spm_processor.decode(gt_caption)
        print('\n')
        print('gt_caption_text: ', gt_caption_text)
        print('gt_caption: ', gt_caption)
        print('pr_caption: ', pred_caption_fwd)

        cv2.imshow('win', img)
        cv2.waitKey(0)
