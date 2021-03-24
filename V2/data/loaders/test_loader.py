from PIL import Image
import numpy as np
import os.path as osp
import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_test(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
    """

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=2*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 unlabel=0, # number of unlabeled examples for all the novel categories.
                 transform=None,
                 load=True,
                 **kwargs
                 ):
        
        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load
        self.unlabel = unlabel

        seed = 112
        random.seed(seed)
        np.random.seed(seed)

        self.Epoch_Exemplar = []
        self.Epoch_Tnovel = []
        if self.unlabel != 0:
            self.Epoch_Unlabel = []
        for i in range(epoch_size):
            Tnovel, Exemplar, Unlabel = self._sample_episode()
            self.Epoch_Exemplar.append(Exemplar)
            self.Epoch_Tnovel.append(Tnovel)
            if self.unlabel != 0:
                self.Epoch_Unlabel.append(Unlabel)

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)
        nKnovel = len(Knovel)
        assert((self.nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)
        nUnlabelPerClass = self.unlabel

        Tnovel = []
        Exemplars = []
        if self.unlabel != 0:
            Unlabels = []
        for Knovel_idx in range(len(Knovel)):
            ids = (nEvalExamplesPerClass + self.nExemplars + nUnlabelPerClass)
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], min(ids,len(self.labels2inds[Knovel[Knovel_idx]])) )

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]
            imgs_emeplars = img_ids[nEvalExamplesPerClass:nEvalExamplesPerClass+self.nExemplars]
            if self.unlabel != 0:
                imgs_unlabel = img_ids[nEvalExamplesPerClass+self.nExemplars:]

            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]
            if self.unlabel != 0:
                Unlabels += [(img_id, Knovel_idx) for img_id in imgs_unlabel]
        assert(len(Tnovel) == self.nTestNovel)
        assert(len(Exemplars) == nKnovel * self.nExemplars)
        random.shuffle(Exemplars)
        random.shuffle(Tnovel)
        if self.unlabel != 0:
            random.shuffle(Unlabels)
            return Tnovel, Exemplars, Unlabels
        else:
            return Tnovel, Exemplars, None

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
        """

        images = []
        labels = []
        for (img_idx, label) in examples:
            img = self.dataset[img_idx][0]
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        return images, labels

    def __getitem__(self, index):
        Tnovel = self.Epoch_Tnovel[index]
        Exemplars = self.Epoch_Exemplar[index]
        if self.unlabel != 0:
            Unlabels = self.Epoch_Unlabel[index]
        Xt, Yt = self._creatExamplesTensorData(Exemplars)
        Xe, Ye = self._creatExamplesTensorData(Tnovel)
        if self.unlabel != 0:
            Xu, _ = self._creatExamplesTensorData(Unlabels)
            return Xt, Yt, Xe, Ye, Xu
        else:
            return Xt, Yt, Xe, Ye, 0
