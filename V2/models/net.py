import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet12 import resnet12


class Model(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0/75, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def get_embeddings(self, x):
        f = self.base(x)
        f = f.mean(2).mean(2)
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        return f


    def test(self, ftrain, ftest):
        ftrain = ftrain.mean(3).mean(3)
        ftest = ftest.mean(3).mean(3)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.bmm(ftest, ftrain.transpose(1,2))
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:]) 
        if not self.training:
            return self.test(ftrain, ftest)

        b, n1, c, h, w = ftrain.size()
        n2 = ftest.size(1)
        ftrain = ftrain.view(b, n1, c, -1) 
        ftest = ftest.view(b, n2, c, -1)
        ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1)
        ftrain = ftrain.view(b, n1, n2, c, h, w).transpose(1, 2)
        ftest = ftest.unsqueeze(1).repeat(1,1,n1,1,1)
        ftest = ftest.view(b, n1, n2, c, h, w).transpose(1, 2)

        ftrain = ftrain.mean(4).mean(4)  

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3) 
        ytest = ytest.unsqueeze(3) 
        ftest = torch.matmul(ftest, ytest) 
        ftest = ftest.view(batch_size * num_test, *f.size()[1:])
        ytest = self.clasifier(ftest)
        return ytest, cls_scores