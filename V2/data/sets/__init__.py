from data.sets.miniImageNet import miniImageNet
from data.sets.tieredImageNet import tieredImageNet
from data.sets.cifarfs import CIFARFS
from data.sets.cub import CUB


__imgfewshot_factory = {
        'miniImageNet': miniImageNet,
        'tieredImageNet': tieredImageNet,
        'cifarfs': CIFARFS,
        'cub': CUB,
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

