import argparse
import os
import random

import numpy as np
import torch
from loguru import logger

import ADMM
from data.data_loader import load_data


def run():
    # Load config
    args = load_config()
    logger.add('logs/{}_model_{}_code_{}.log'.format(
        args.dataset,
        args.arch,
        args.code_length,

        0.01,
        #args.alpha,

    ),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    train_dataloader, query_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.batch_size,
        args.num_workers,
    )

    # Training
    for code_length in args.code_length:
        checkpoint = ADMM.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader, 
            args.arch,
            code_length,
            args.device,
            args.lr,
            args.max_iter,
            args.topk,
            args.evaluate_interval,
            args.anchor_num,
            args.proportion,
        )
        logger.info('[code_length:{}][map:{:.4f}]'.format(args.code_length, checkpoint['map']))

        # Save checkpoint
        torch.save(
            checkpoint,
            os.path.join('checkpoints', '{}_model_{}_code_{}_alpha_{}_map_{:.4f}.pt'.format(
                args.dataset,
                args.arch,
                code_length,
                0.01,
                #args.alpha,
                checkpoint['map']),
                         )
        )


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Prototypical Semantic Hashing.')
    parser.add_argument('--dataset', default='cifar-10',
                        help='Dataset name.')
    parser.add_argument('--root', default='data/cifar-10',
                        help='Path of dataset')
    parser.add_argument('--code-length', default='16,32,48,64', type=str,
                        help='Binary hash code length. (default: 16,32,48,64)')
    parser.add_argument('--arch', default='vgg16', type=str,
                        help='CNN model name.(default: vgg16)')#alexnet
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Batch size.(default: 8)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--max-iter', default=3, type=int,
                        help='Number of iterations.(default: 100)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=1, type=int,
                        help='Evaluation interval.(default: 10)')
    parser.add_argument('--anchor_num', default=100, type=int,
                        help='Number of anchors.(default: 100)')
    parser.add_argument('--proportion', default=0.999, type=float,
                        help='Number of anchors.(default: 0.999)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        print("as you wish")
        #args.device = torch.device("cpu")
        args.device = torch.device("cuda:%d" % 0)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.device = torch.device("cuda:%d" % 0)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':
    run()
