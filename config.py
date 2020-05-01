import argparse


def config():
    parser = argparse.ArgumentParser('ICI-FSL')
    parser.add_argument('--folder', type=str, default='data',
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to the trained model.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--classifier', type=str, default='lr',
                        help='lr/svm.')
    parser.add_argument('--gpu', '-g', type=str, default='0')
    parser.add_argument('--mode', type=str, default='test',
                        help='train/test')
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--num_shots', type=int, default=1,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num_test_ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--step', type=int, default=5,
                        help='Select how many unlabeled data for each class in one iteration.')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate for training the feature extractor.')
    parser.add_argument('--output-folder', type=str, default='./ckpt',
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--num-batches', type=int, default=600,
                        help='Number of batches for test(default: 600).')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--dim', type=int, default=5,
                        help='Reduced dimension.')
    parser.add_argument('--embed', type=str, default='pca',
                        help='Dimensionality reduction algorithm.')
    parser.add_argument('--unlabel', type=int, default=0,
                        help='Number of unlabeled examples per class, 0 means TFSL setting.')
    parser.add_argument('--img_size', type=int, default=84)
    args = parser.parse_args()
    return args
