import argparse

def config():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet')
    parser.add_argument('--load', default=False)

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=90, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--LUT_lr', default=[(60, 0.1), (70, 0.006), (80, 0.0012), (90, 0.00024)],
                        help="multistep to decay learning rate")

    parser.add_argument('--train-batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=1, type=int,
                        help="test batch size")

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--scale_cls', type=int, default=7)

    # ************************************************************
    # ICI parameters
    # ************************************************************
    parser.add_argument('--classifier', type=str, default='lr')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='linear')
    parser.add_argument('--embed', type=str, default='lle')
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--logit_penalty', type=float, default=0.5)

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('-g', '--gpu-devices', default='0', type=str)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')
    parser.add_argument('--unlabel', type=int, default=0)

    parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='test', type=str,
                        help='use test or val dataset to early stop')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    # Dataset-related parameters
    if args.dataset == 'miniImageNet':
        args.num_classes = 64
        args.load = True
        args.dataset_dir = '/home/wyk/data/ici/miniImageNet'
    elif args.dataset == 'tieredImageNet':
        args.num_classes = 351
        args.train_epoch_size = 13980
        args.max_epoch = 90
        args.LUT_lr = [(30, 0.1), (60, 0.01), (90, 0.001)]
        args.dataset_dir = '/home/wyk/data/ici/tieredImageNet'
    elif args.dataset == 'cifarfs':
        args.num_classes = 64
        args.dataset_dir = '/home/wyk/data/ici/cifarfs/images'
    elif args.dataset == 'cub':
        args.num_classes = 100
        args.dataset_dir = '/home/wyk/data/ici/cub/images'
    else:
        raise NameError

    return args

