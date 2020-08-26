"""Normalizing medical concepts using pre-trained Roberta Model. We also use
SNOMED synonyms as additional training examples.

@author: Nikhil Pattisapu, iREL, IIIT-H"""


# pylint: disable=import-error, no-member
import os
import pickle
import argparse
from random import shuffle
from pprint import pprint
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (RobertaTokenizer, RobertaConfig,
                          RobertaForSequenceClassification)



def get_best_match(ph_rep, snomed_rep, snomed_ids, snomed_desc, n_res):
    '''Returns top "n" medical concepts per paraphrase'''
    mapped_ids, mapped_desc = [], []
    snomed_ids, snomed_desc = np.array(snomed_ids), np.array(snomed_desc)
    # Get cosine similarity between phrase vectors and SNOMED concept
    # descriptions.
    val = cosine_similarity(ph_rep, snomed_rep)

    # Get the indices corresponding to top 'n_res' cosine sim. values.
    # NOTE: There is no ordering in the top 'n_res' values.
    for idx in range(val.shape[0]):
        top_ind = np.argpartition(val[idx], -n_res)[-n_res:]
        top_val = val[idx][top_ind]

        # Sort the top 'n_res' values. Default sort order: smallest first.
        # Reverse this list to get highest cosine similarity.
        top_val, top_ind = zip(*sorted(zip(top_val, top_ind)))
        top_ind = list(reversed(top_ind))
        mapped_ids.append(list(snomed_ids[top_ind]))
        mapped_desc.append(list(snomed_desc[top_ind]))
    return mapped_ids, mapped_desc


def get_nearest_label(ph_rep):
    """Return the labels given the representations using cosine similarity"""
    sids, sdesc = get_best_match(ph_rep, REPR, SIDS, DESC, n_res=1)
    sids = [sid[0] for sid in sids]
    sdesc = [desc[0] for desc in sdesc]
    return sids, sdesc


def filter_set(texts, labels):
    """Filter out examples where labels are not present"""
    assert len(texts) == len(labels)
    sid_set = set(SIDS)
    n_txt, n_lbl = [], []
    for txt, lbl in zip(texts, labels):
        if lbl in sid_set:
            n_txt.append(txt)
            n_lbl.append(lbl)
    return n_txt, n_lbl


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **_kwargs):
        """Forward function"""
        # pylint: disable=invalid-name
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def train_model_get_pred(tr_txt, tr_lab, te_txt, te_lab, lbl_ix):
    """ Returns trained model"""
    # Load pretrained stuff and initialize the classifier

    pad = ' '.join(['<pad>'] * MAX_LEN)
    pad = ' ' + pad
    tr_txt = [item.strip() + pad for item in tr_txt]
    tr_lab = [item.strip() for item in tr_lab]
    te_txt = [item.strip() + pad for item in te_txt]
    ix_lbl = {v: k for k, v in lbl_ix.items()}

    cls = RobertaForSequenceClassification.from_pretrained('roberta-base')
    cls.num_labels = len(ix_lbl)
    config = RobertaConfig()
    config.num_labels = SID_EMB_DIM
    custom_classifier = RobertaClassificationHead(config)
    cls.classifier = custom_classifier
    cls = cls.to(DEVICE)

    opt = torch.optim.AdamW(cls.parameters(), lr=2e-5)
    criteria = torch.nn.CosineEmbeddingLoss()

    n_tr, n_te = len(tr_txt), len(te_txt)
    data = list(zip(tr_txt, tr_lab))
    te_txt = [te_txt[i: i + N_BATCH] for i in range(0, n_te, N_BATCH)]
    for _epoch in tqdm(range(MAX_EPOCHS)):
        shuffle(data)  # Shuffle the data every epoch.
        tr_txt = [item[0] for item in data]
        tr_lab = [item[1] for item in data]

        # Batchify the texts and labels
        tr_txt = [tr_txt[i: i + N_BATCH] for i in range(0, n_tr, N_BATCH)]
        tr_lab = [tr_lab[i: i + N_BATCH] for i in range(0, n_tr, N_BATCH)]
        tr_loss = 0
        for b_txt, b_lab in list(zip(tr_txt, tr_lab)):
            cls.train()
            opt.zero_grad()
            x = [tok.encode(t, add_special_tokens=True)[:MAX_LEN]
                 for t in b_txt]
            x = torch.LongTensor(x).to(DEVICE)
            y_tr = np.vstack([SID_EMB[sid] for sid in b_lab])
            y_tr = torch.from_numpy(y_tr).float().to(DEVICE)
            y_pr = cls(x)[0].float()
            one_tensor = torch.ones(y_tr.shape[0]).to(DEVICE)
            loss = criteria(y_pr, y_tr, one_tensor)
            loss.backward()
            tr_loss += loss.item()
            opt.step()

        # print('The training loss for {} epoch is {}'.format(epoch, tr_loss))

    te_prd = []
    with torch.no_grad():
        for b_txt in te_txt:
            cls.eval()
            x = [tok.encode(t, add_special_tokens=True)[:MAX_LEN]
                 for t in b_txt]
            x = torch.LongTensor(x).to(DEVICE)
            y_pr = cls(x)[0]
            y_pr = y_pr.detach().cpu().numpy()
            pr_sid, _pr_desc = get_nearest_label(y_pr)
            te_prd += pr_sid
    acc = accuracy_score(te_lab, te_prd)
    pprint(acc)
    return te_prd


def print_mcn_acc(dir_name, n_folds):
    """The main function"""
    files = os.listdir(dir_name)
    files = [item for item in files if '.tsv' in item]
    labels = [l.strip() for l in open(dir_name + '/labels.txt',
                                      'r').readlines()]
    lbl_ix = {lbl: ix for ix, lbl in enumerate(labels)}

    # Pretrain with SNOMED data
    snomed = open(dir_name + 'snomed.txt', 'r').readlines()
    snomed_txt, snomed_lbl = [], []
    for line in snomed:
        parts = line.split('\t')
        snomed_txt.append(parts[0].strip())
        snomed_lbl.append(parts[1].strip())

    for ix in range(n_folds):
        print('Current fold is {}'.format(ix))
        tr = [f for f in files if 'train' in f and str(ix) in f][0]
        te = [f for f in files if 'test' in f and str(ix) in f][0]
        tr = dir_name + tr
        te = dir_name + te
        tr_lines = [l.strip().split('\t') for l in open(tr, 'r').readlines()]
        te_lines = [l.strip().split('\t') for l in open(te, 'r').readlines()]

        tr_txt = [item[0].strip() for item in tr_lines]
        tr_lab = [item[1].strip() for item in tr_lines]
        te_txt = [item[0].strip() for item in te_lines]
        te_lab = [item[1].strip() for item in te_lines]

        tr_txt = tr_txt + snomed_txt
        tr_lab = tr_lab + snomed_lbl

        tr_txt, tr_lab = filter_set(tr_txt, tr_lab)
        te_txt, te_lab = filter_set(te_txt, te_lab)

        te_prd = train_model_get_pred(tr_txt, tr_lab, te_txt, te_lab, lbl_ix)
        print(accuracy_score(te_lab, te_prd))
        # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True) # target emb pkl path
    parser.add_argument('--desc', type=str, required=True) # SNOMED desc pkl path
    parser.add_argument('--data', type=str, required=True) # Dataset path
    args = parser.parse_args()
    # pylint: disable=arguments-differ, invalid-name, too-many-locals
    # HOME = '/home/nikhil.pattisapu/projects/unbounded_mcn/'
    BERT_DIM = 768
    N_BATCH = 16
    MAX_LEN = 15
    MAX_EPOCHS = 70
    DEVICE = torch.device('cuda:0')

    # SID_DESC_PATH = HOME + 'resources/sid_to_desc.pkl'
    # SID_EMB_PATH = HOME + 'resources/sid_to_elmo.pkl'
    # SID_DESC_PATH = HOME + 'resources/sid_to_desc.pkl'
    SID_EMB = pickle.load(open(args.target, 'rb'))
    SID_DESC = pickle.load(open(args.desc, 'rb'))
    SID_EMB = {str(k): v for k, v in SID_EMB.items()}
    SID_DESC = {str(k): str(v) for k, v in SID_DESC.items()}
    SID_EMB_DIM = SID_EMB['22298006'].shape[0]  # Embedding for heart attack.
    SIDS = [ix.strip() for ix in open(args.data + '/labels.txt', 'r').readlines()]
    SIDS = [sid for sid in SIDS if sid in SID_EMB]
    DESC = [SID_DESC[sid] for sid in SIDS]
    REPR = np.vstack([SID_EMB[sid] for sid in SIDS])
    tok = RobertaTokenizer.from_pretrained('roberta-base')
    num_folds = 5
    print_mcn_acc(args.data, num_folds)
