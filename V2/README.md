# How to Trust Unlabeled Data? Instance Credibility Inference for Few-Shot Learning

\[[paper](https://arxiv.org/abs/2007.08461)\]

This repository contains the code for our paper "How to Trust Unlabeled Data? Instance Credibility Inference for Few-Shot Learning". 

## Requirements

python=3.7.6

torch=1.5.1

sklearn=0.23.2

glmnet-py=0.1.0b2

tqdm

## DataSet Preparation

You can download the dataset from [miniImageNet](https://github.com/gidariss/FewShotWithoutForgetting), [tieredImageNet](https://github.com/yaoyao-liu/meta-transfer-learning), [CIFAR-FS](https://github.com/bertinetto/r2d2), [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), respectively.
Or you can use your own dataet and modify the corresponding python file in *data/sets/*.

Please note that our experiments on CUB using the images *cropped* by the provided bounding box.

## Pretrained Models

Pretrained models are in the folder *ckpt*.

## Training

If you want to train the feature extractor from scratch, you should first check the data related hyper-parameters in *config.py*.

Then, to train the feature extractor in 5-way-1-shot task of miniImageNet you should run:

```
python main.py --dataset miniImageNet --save-dir ckpt/miniImageNet/1-shot -g 0 --nKnovel 5 --nExemplars 1 --phase val --mode train
```
If you want to train on 5-shot task, using the option:
```
--nExemplars 5
```

## Testing

If you want to test on TFSL setting, run:
```
python main.py --dataset miniImageNet --save-dir ckpt/miniImageNet/test -g 0 --nKnovel 5 --nExemplars 1 --phase test --mode test --resume $MODEL_PATH
```
To test with the logistic regression version of ICI, add the option:
```
--strategy logit
```
To test on the SSFSL setting, add the option:
```
--unlabel $NUM_OF_UNLABEL
```
where $NUM_OF_UNLABEL indicate the number of unlabeled data for each class.

For other options, please check *config.py*.

## Acknowledgments

This code is based on [CAN](https://github.com/blue-blue272/fewshot-CAN) and [MetaOptNet](https://github.com/kjunelee/MetaOptNet).


## Citation

If you found the provided code useful, please cite our work.

```
@article{wang2020trust,
  title={How to trust unlabeled data? Instance Credibility Inference for Few-Shot Learning},
  author={Wang, Yikai and Zhang, Li and Yao, Yuan and Fu, Yanwei},
  journal={arXiv preprint arXiv:2007.08461},
  year={2020}
}
```

