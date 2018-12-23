import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class S2S(nn.Module):
    def __init__(self, embed_mat):
        super(S2S, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, self.vocab_num))

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        h1, h1_n = self.encode(x)
        del h1
        h2, h2_n = self.decode(y, h1_n)
        return self.dl(h2)


class S2SEncode(nn.Module):
    def __init__(self, embed_mat):
        super(S2SEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        del h1
        return h1_n


class S2SDecode(nn.Module):
    def __init__(self, embed_mat):
        super(S2SDecode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, self.vocab_num))

    def forward(self, y, h1_n):
        y = self.embed(y)
        h2, h2_n = self.decode(y, h1_n)
        return self.dl(h2)


class Att(nn.Module):
    def __init__(self, embed_mat):
        super(Att, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, self.vocab_num))

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        return self.dl(s2)


class AttEncode(nn.Module):
    def __init__(self, embed_mat):
        super(AttEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        return h1


class AttDecode(nn.Module):
    def __init__(self, embed_mat):
        super(AttDecode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, self.vocab_num))

    def forward(self, y, h1):
        y = self.embed(y)
        h1_n = torch.unsqueeze(h1[:, -1, :], dim=0)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        return self.dl(s2)


class AttPlot(nn.Module):
    def __init__(self, embed_mat):
        super(AttPlot, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        return F.softmax(d, dim=-1)
