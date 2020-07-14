# Instance Credibility Inference for Few-Shot Learning

\[[paper](https://arxiv.org/pdf/2003.11853.pdf)\]\[[intro](https://yikai-wang.github.io/ici)\]\[[中文介绍](https://zhuanlan.zhihu.com/p/120354934)\]

This repository contains the code for our paper "Instance Credibility Inference for Few-Shot Learning". 

## Requirements

python=3.7.3

torch=1.1.0

sklearn=0.21.2

tqdm

*Note:* We have found the existence of randomness when running the code under different servers even we use the same version of packages. Hence the accuracy may rise or fall by about 1% compared with our reported performance. 

## Usage

The image path of each dataset is save in the corresponding *train.csv*, *valid.csv*, *test.csv* files, you may create the list to match the dataset in your machine or re-write the *DataSet* and *EmbeddingDataset* class in the *datasets.py* based on your data.

When training and testing, you need to set some hyperparameters. For example:

```
python main.py -g 2 --resume ckpt/res12_mini.pth.tar --dataset miniimagenet
```

Then the program will output the following:

```
Namespace(ckpt=None, classifier='lr', dataset='miniimagenet', device=device(type='cuda', index=0), dim=5, embed='pca', folder='data', gpu='2', img_size=84, lr=0.1, mode='test', num_batches=600, num_shots=1, num_test_ways=5, num_workers=4, output_folder='./ckpt', resume='ckpt/res12_mini.pth.tar', step=5, unlabel=0)
100% 600/600 [37:30<00:00,  1.86s/it]
Test Acc Mean56.06 65.32 66.74 66.80 66.80
Test Acc ci0.773 1.010 1.084 1.096 1.097
```

where the first line is all the hyper-parameters, the second line is the running time reported by *tqdm*. The following two lines report the mean and confidence interval of test accuracy in each step. In our experiments, we use the results of different steps under different setting, please check our paper for details.

If you want to train the embedding network, use the command:

```
--mode train
```

If you want to test the performance, use the command:

```
--mode test
```

If you want to test under the SSFSL setting, use the command (use 15 unlabeled images for each category, and you can select a number you want):

```
--unlabel 15
```

Set the number of shots, use:

```
--num_shots 1
```

Set the number of ways, use:

```
--num_test_ways 5
```

For all the options, please check the *config.py*.

## FAQs

*How to create the csv files?*

In the csv file, each line is related to one image where the first column is the path and the second column is the category/label. You may check/modify the following code:

https://github.com/Yikai-Wang/ICI-FSL/blob/0d6a3e5b3403a8a3d8f22b04f91406fa4650fd97/datasets.py#L19-L35

## Citation

If you found the provided code useful, please cite our work.

```
@inproceedings{wang2020instance,
  title={Instance Credibility Inference for Few-Shot Learning},
  author={Wang, Yikai and Xu, Chengming and Liu, Chen and Zhang, Li and Fu, Yanwei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

