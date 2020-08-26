""" Train MCN on snomed + weakly supervised data and evaluate on cadec, psytar
and twadr.

@author: Nikhil Pattisapu, iREL, IIIT-H"""

import sys
import pickle
import copy
from random import shuffle
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaConfig
from transformers import RobertaForSequenceClassification


# Define global variables including dataset and embedding
# pylint: disable=invalid-name, no-member, too-many-locals, arguments-differ

# File paths
HOME = '/home/nikhil/OneDrive/ws_mcn/'
DATA_PATH = HOME + 'data/cadec_62/'
SID_EMB_PATH = HOME + 'resources/sid_to_transE.pkl'
SID_DESC_PATH = HOME + 'resources/sid_to_desc.pkl'
SID_LABEL_PATH = DATA_PATH + 'labels.txt'
WS_PATH = DATA_PATH + 'ws_uniq.txt'
SNOMED_PATH = DATA_PATH + 'snomed.txt'

# Load resources
SID_EMB = pickle.load(open(SID_EMB_PATH, 'rb'))
SID_EMB = {str(k): v for k, v in SID_EMB.items()}
SID_DESC = pickle.load(open(SID_DESC_PATH, 'rb'))
SID_DESC = {str(k): v for k, v in SID_DESC.items()}
SIDS = [sid.strip() for sid in open(SID_LABEL_PATH, 'r').readlines()]
SIDS = [sid for sid in SIDS if sid in SID_EMB]
DESC = [SID_DESC[sid] for sid in SIDS]
REPR = np.vstack([SID_EMB[sid] for sid in SIDS])

# Hyperparameter values
BERT_DIM = 768
N_BATCH = 16
MAX_LEN = 15
MAX_EPOCHS = 70
SID_EMB_DIM = SID_EMB['22298006'].shape[0]  # Get embedding for heart attack.

DEVICE = torch.device('cuda:0')
TOK = RobertaTokenizer.from_pretrained('roberta-base')


def get_best_concepts(ph_rep, snomed_rep, snomed_ids, snomed_desc, n_res):
    '''Returns top "n" medical concepts per paraphrase'''
    mapped_ids, mapped_desc = [], []
    snomed_ids, snomed_desc = np.array(snomed_ids), np.array(snomed_desc)
    # Get cosine similarity between phrases and concepts
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


def get_ws_phrases_kbest(n_res):
    '''Returns the nearest "n" phrases for a given medical concept'''
    ws_phrases, ws_lbls = [], []
    ws_df = pd.read_csv(WS_PATH, sep='\t', lineterminator='\n')
    snomed_group = ws_df.groupby('snomed_id', sort=False)
    for sid, sdf in snomed_group:
        sdf = sdf.sort_values(['similarity'], ascending=False)
        sdf = sdf[sdf['similarity'] < 0.90]
        sdf = sdf.drop_duplicates().head(n_res)
        nearest = list(sdf['phrase'])
        labels = [sid] * len(nearest)
        ws_phrases += nearest
        ws_lbls += labels
    return ws_phrases, ws_lbls


def get_nearest_label(ph_rep):
    """Return the labels given the representations using cosine similarity"""
    sids, sdesc = get_best_concepts(ph_rep, REPR, SIDS, DESC, n_res=1)
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
        """Feed forward operation"""
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def get_mcn_model():
    """Initialize an MCN model"""
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    config = RobertaConfig()
    config.num_labels = SID_EMB_DIM
    custom_classifier = RobertaClassificationHead(config)
    model.classifier = custom_classifier
    model = model.to(DEVICE)
    return model


def preprocess(txt, lbl):
    """Preprocessing involves padding, shuffling and batch"""
    assert len(txt) == len(lbl)
    pad = ' '.join(['<pad>'] * MAX_LEN)
    txt = [item.strip().lower() + pad for item in txt]
    lbl = [str(item).strip().lower() for item in lbl]
    data = list(zip(txt, lbl))
    shuffle(data)  # Shuffle the data every epoch.

    n_txt, n_lbl = [], []
    for t, l in data:
        if l in SID_EMB:
            n_txt.append(t)
            n_lbl.append(l)

    n_samples = len(n_txt)
    txt = [n_txt[i: i + N_BATCH] for i in range(0, n_samples, N_BATCH)]
    lbl = [n_lbl[i: i + N_BATCH] for i in range(0, n_samples, N_BATCH)]
    return txt, lbl


def train_model(tr_txt, tr_lbl, model, n_epochs):
    """Returns trained model"""
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criteria = torch.nn.CosineEmbeddingLoss()

    for _epoch in tqdm(range(n_epochs)):
        tr_txt_batches, tr_lbl_batches = preprocess(tr_txt, tr_lbl)
        tr_loss = 0
        for b_txt, b_lbl in list(zip(tr_txt_batches, tr_lbl_batches)):
            model.train()
            opt.zero_grad()
            x = [TOK.encode(t, add_special_tokens=True)[:MAX_LEN] for t in b_txt]
            x = torch.LongTensor(x).to(DEVICE)
            y_tr = np.vstack([SID_EMB[sid] for sid in b_lbl])
            y_tr = torch.from_numpy(y_tr).float().to(DEVICE)
            y_pr = model(x)[0].float()
            one_tensor = torch.ones(y_tr.shape[0]).to(DEVICE)
            loss = criteria(y_pr, y_tr, one_tensor)
            loss.backward()
            tr_loss += loss.item()
            opt.step()
        # print('The training loss for {} epoch is {}'.format(_epoch, tr_loss))
    return model


def get_predictions(te_txt, te_lbl, model):
    """ Returns predictions and accuracy"""
    # Load pretrained stuff and initialize the classifier
    te_txt, te_lbl = preprocess(te_txt, te_lbl)
    te_prd = []
    with torch.no_grad():
        for b_txt in te_txt:
            model.eval()
            x = [TOK.encode(t, add_special_tokens=True)[:MAX_LEN] for t in b_txt]
            x = torch.LongTensor(x).to(DEVICE)
            y_pr = model(x)[0]
            y_pr = y_pr.detach().cpu().numpy()
            pr_sid, _pr_desc = get_nearest_label(y_pr)
            te_prd += pr_sid
    te_lbl = [item for sublist in te_lbl for item in sublist]
    acc = accuracy_score(te_lbl, te_prd)
    return te_prd, acc


def get_cv_acc(pretrained_model, n_folds, training=True):
    """Returns cross validation accuracy"""
    cv_acc = 0
    for ix in range(n_folds):
        if training:
            model = copy.deepcopy(pretrained_model)
            df_tr = pd.read_csv(DATA_PATH + 'train_' + str(ix) + '.tsv',
                                sep='\t', lineterminator='\n', header=None)
            tr_txt, tr_lbl = df_tr[0].tolist(), df_tr[1].tolist()
            model = train_model(tr_txt, tr_lbl, model, MAX_EPOCHS)
        else:
            model = pretrained_model
        df_te = pd.read_csv(DATA_PATH + 'test_' + str(ix) + '.tsv',
                            sep='\t', lineterminator='\n', header=None)
        te_txt, te_lbl = df_te[0].tolist(), df_te[1].tolist()
        _te_prd, acc = get_predictions(te_txt, te_lbl, model)
        cv_acc += acc
    return cv_acc / n_folds


if __name__ == '__main__':
    model_init = get_mcn_model()
    n_folds = 5
    df = pd.read_csv(SNOMED_PATH, sep='\t', lineterminator='\n', header=None)
    snm_txt, snm_lbl = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
    model_snm = train_model(snm_txt, snm_lbl, model_init, n_epochs=70)
    snm_acc = get_cv_acc(model_snm, n_folds=n_folds, training=False)
    print(snm_acc)

    ws_txt, ws_lbl = get_ws_phrases_kbest(5)
    model_snm_ws = train_model(ws_txt, ws_lbl, model_snm, n_epochs=50)
    snm_ws_acc = get_cv_acc(model_snm_ws, n_folds=n_folds, training=False)
    print(snm_ws_acc)

    model_snm_ws_snm = train_model(snm_txt, snm_lbl, model_snm_ws, n_epochs=30)
    snm_ws_snm_acc = get_cv_acc(model_snm_ws_snm, n_folds=n_folds,
                                training=False)
    print(snm_ws_snm_acc)
