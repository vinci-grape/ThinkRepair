import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)
    
    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),output_hidden_states=True)

        token_embeddings = out[0]

        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege
        sentence_embeddings = sentence_embeddings

        return sentence_embeddings

    def forward(self, input_ids):
        
        vec = self.get_xcode_vec(input_ids)
        logits = self.classifier(vec)
    
        # keep dropout twice
        vec2 = self.get_xcode_vec(input_ids)
        logits2 = self.classifier(vec2)
        
        loss = self.compute_kl_loss(logits, logits2)
        
        return loss


    def compute_kl_loss(self, p, q, pad_mask=None):
        
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    

    def simcse_unsup_loss(self, vec, device, temp=0.05):

        # label: [0, batch_size], [1, batch_size+1],....,[batch_size-1, 2*batch_size-1]
        label = torch.arange(vec.shape[0], device=device)
        label = torch.roll(label, shifts=int(len(label)/2))

        # [batch_size*2, 1, 768] * [1, batch_size*2, 768] = [batch_size*2, batch_size*2]
        sim = F.cosine_similarity(vec.unsqueeze(1), vec.unsqueeze(0), dim=-1)

        sim = sim - torch.eye(vec.shape[0], device=device) * 1e12
        sim = sim / temp

        # cross entropy loss for similiarity
        loss = F.cross_entropy(sim, label)
        return torch.mean(loss)