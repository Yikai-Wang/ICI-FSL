import math
import os
import sys

import glmnet_python
import numpy as np
import scipy
from glmnet import glmnet
from glmnetCoef import glmnetCoef
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ICI(object):

    def __init__(self, classifier='lr', num_class=None, step=5, max_iter='auto',
                 reduce='lle', d=5, norm='l2', strategy='linear',logit_penalty=0.5):
        self.step = step
        self.max_iter = max_iter
        self.num_class = num_class
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.initial_classifier(classifier)
        self.initial_strategy(strategy)
        self.strategy = strategy
        self.logit_penalty = logit_penalty

    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y


    def predict(self, X, unlabel_X=None, show_detail=False, query_y=None):
        support_X, support_y = self.support_X, self.support_y
        way, num_support = self.num_class, len(support_X)
        query_X = self.norm(X)
        if unlabel_X is None:
            unlabel_X = query_X
        else:
            unlabel_X = self.norm(unlabel_X)
        num_unlabel = unlabel_X.shape[0]
        assert self.support_X is not None
        embeddings = np.concatenate([support_X, unlabel_X])
        X = self.embed(embeddings)
        if 'logit' not in self.strategy:
            H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
            X_hat = np.eye(H.shape[0]) - H
        else:
            I = np.eye((X.shape[0]))
            X_hat = np.concatenate([X, I],axis=1)
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabel/self.step)
        else:
            assert float(self.max_iter).is_integer()
        support_set = np.arange(num_support).tolist()
        self.classifier.fit(self.support_X, self.support_y)
        if show_detail:
            acc_list = []
        for _ in range(self.max_iter):
            if show_detail:
                predicts = self.classifier.predict(query_X)
                acc_list.append(np.mean(predicts == query_y))
            pseudo_y = self.classifier.predict(unlabel_X)
            y = np.concatenate([support_y, pseudo_y])
            Y = self.label2onehot(y, way)
            if 'logit' not in self.strategy:
                y_hat = np.dot(X_hat, Y)
            else:
                y_hat = Y
            support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y)
            y = np.argmax(Y, axis=1)
            self.classifier.fit(embeddings[support_set], y[support_set])
            if len(support_set) == len(embeddings):
                break
        predicts = self.classifier.predict(query_X)
        if show_detail:
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts

    def initial_strategy(self, strategy):
        if strategy == 'linear':
            self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                         normalize=True, warm_start=True, selection='cyclic')
            self.expand = self.linear
        elif strategy == 'logit':
            self.expand = self.logit
        else:
            raise NameError

    def logit(self, support_set, X_hat, y_hat, way, num_support, pseudo_y):
        pfac = scipy.ones([X_hat.shape[1],5])
        d = X_hat.shape[1]-X_hat.shape[0]
        pfac[:d,:]= self.logit_penalty
        X_hat = X_hat.astype('float')
        y_hat = y_hat.astype('float')
        with HiddenPrints():
            """ we use this to avoid print informations like 
                "Warning: Non-fatal error in glmnet library call: error code =  -44
                Check results for accuracy. Partial or no results returned."
                The default length of coefficient is 100. 
                When the warning information prints, the length is smaller than 100,
                which is not a fatal problem in our algorithm, so we simply ignore it.
                Potential lengths in our experiments when the warning happens are 43, 45, 46, etc.
            """
            clf = glmnet(x=X_hat.copy(), y=y_hat.copy(), alpha=1.0, family='multinomial', mtype='grouped', penalty_factor=pfac)
        lambdas = clf['lambdau'][::-1]
        selected = np.zeros(way)
        for lambd in lambdas:
            gamma = glmnetCoef(clf, s=scipy.float64([lambd]), exact=False)
            gamma = np.array(gamma)
            gamma = gamma.reshape(gamma.shape[:-1])[:, 1:].T
            gamma = gamma[d:]
            for i, g in enumerate(gamma[num_support:]):
                if (np.sum(np.abs(g))) == 0.0 and \
                    (i+num_support not in support_set) and \
                        (selected[pseudo_y[i]] < self.step):
                    support_set.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
        return support_set

    def linear(self, support_set, X_hat, y_hat, way, num_support, pseudo_y):
        _, coefs, _ = self.elasticnet.path(X_hat, y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[
                       ::-1, num_support:, :]), axis=2)
        selected = np.zeros(way)
        for gamma in coefs:
            for i, g in enumerate(gamma):
                if g == 0.0 and \
                    (i+num_support not in support_set) and \
                        (selected[pseudo_y[i]] < self.step):
                    support_set.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
        return support_set


    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['isomap', 'ltsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'ltsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm', 'knn']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        elif classifier == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier(n_neighbors=1)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
