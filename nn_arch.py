import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S2S(nn.Module):
    def __init__(self, embed_mat):
        super(S2S, self).__init__()
        self.encode = S2SEncode(embed_mat)
        self.decode = S2SDecode(embed_mat)

    def forward(self, x, y):
        h1_n = self.encode(x)
        return self.decode(y, h1_n)


class S2SEncode(nn.Module):
    def __init__(self, embed_mat):
        super(S2SEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.encode = nn.GRU(embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        return h1_n


class S2SDecode(nn.Module):
    def __init__(self, embed_mat):
        super(S2SDecode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, vocab_num))

    def forward(self, y, h1_n):
        y = self.embed(y)
        h2, h2_n = self.decode(y, h1_n)
        return self.dl(h2)


class Att(nn.Module):
    def __init__(self, embed_mat):
        super(Att, self).__init__()
        self.encode = AttEncode(embed_mat)
        self.decode = AttDecode(embed_mat)

    def forward(self, x, y):
        h1 = self.encode(x)
        return self.decode(y, h1)


class AttEncode(nn.Module):
    def __init__(self, embed_mat):
        super(AttEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.encode = nn.GRU(embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        return h1


class AttDecode(nn.Module):
    def __init__(self, embed_mat):
        super(AttDecode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
        self.qry = nn.Linear(200, 200)
        self.key = nn.Linear(200, 200)
        self.val = nn.Linear(200, 200)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, vocab_num))

    def forward(self, y, h1):
        y = self.embed(y)
        h1_n = torch.unsqueeze(h1[:, -1, :], dim=0)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.qry(h2), self.key(h1), self.val(h1)
        d = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        return self.dl(s2)


class AttCore(nn.Module):
    def __init__(self, embed_mat):
        super(AttCore, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
        self.qry = nn.Linear(200, 200)
        self.key = nn.Linear(200, 200)
        self.val = nn.Linear(200, 200)

    def forward(self, y, h1):
        y = self.embed(y)
        h1_n = torch.unsqueeze(h1[:, -1, :], dim=0)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.qry(h2), self.key(h1), self.val(h1)
        d = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        return F.softmax(d, dim=-1)
