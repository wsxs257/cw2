import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')

    # Add data arguments
    parser.add_argument('--data', default='europarl_prepared', help='path to data directory')
    parser.add_argument('--source-lang', default='de', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=10, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')

    # Add model arguments
    parser.add_argument('--arch', default='lstm', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=100, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--patience', default=10, type=int,
                        help='number of epochs without improvement on validation set before early stopping')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    """ Main training function. Trains the translation model over the course of several epochs, including dynamic
    learning rate adjustment and gradient clipping. """
    logging.info('Commencing training!')
    torch.manual_seed(42)
    np.random.seed(42)

    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_dict=src_dict, tgt_dict=tgt_dict)

    train_dataset = load_data(split='train') if not args.train_on_tiny else load_data(split='tiny_train')
    valid_dataset = load_data(split='valid')

    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)

    logging.info('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
    criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')

    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer)  # lr_scheduler
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = \
            torch.utils.data.DataLoader(train_dataset, num_workers=1, collate_fn=train_dataset.collater,
                                        batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                   0, shuffle=True, seed=42))
        model.train()
        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = 0
        stats['num_tokens'] = 0
        stats['batch_size'] = 0
        stats['grad_norm'] = 0
        stats['clip'] = 0
        # Display progress
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

        # Iterate over the training set
        for i, sample in enumerate(progress_bar):

            if len(sample) == 0:
                continue
            model.train()

            # sample = sample.to(device)

            '''
            ___QUESTION-1-DESCRIBE-E-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  Add line-by-line description about the following lines of code do.
            '''

            # output.size = [batch_size, tgt_time_steps, vocab_size]
            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            # This line passes a batch of source tokens, their lengths,
            # and target input tokens to the model. The model then generates
            # predictions for the next tokens in the sequence.

            loss = \
                criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1)) / len(sample['src_lengths'])
            # compute the loss value by comparing the predicted output sequence with the target sequence using the
            # criterion function. The loss is then averaged over the number of source sequences in the batch
            loss.backward()
            # This operation computes the gradients of the loss function with respect to the model parameters.
            # It's the step where the model learns,
            # by understanding how to adjust its weights to minimize the loss.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            # Gradient clipping is applied here to prevent exploding gradients,
            # a problem that can lead to wildly oscillating weights and model instability.
            # args.clip_norm specifies the maximum allowed L2-norm of the gradients (default threshold = 4.0).
            # If the norm of the gradients exceeds this value, they will be scaled down proportionally.
            optimizer.step()
            # This method updates the model's parameters based on the gradients computed during backpropagation.
            # It essentially applies the optimization algorithm (e.g., Adam, SGD) to modify the parameters in
            # the direction that minimizes the loss.
            optimizer.zero_grad()
            # Before the next iteration or batch of training,
            # the gradients are reset to zero. This is crucial
            # because gradients accumulate by default in PyTorch,
            # and failing to reset them would result in mixing gradient information across batches.
            '''___QUESTION-1-DESCRIBE-E-END___'''

            # Update statistics for progress bar
            total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
            stats['loss'] += total_loss * len(sample['src_lengths']) / sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        # Calculate validation loss
        valid_perplexity = validate(args, model, criterion, valid_dataset, epoch)
        model.train()

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_perplexity)  # lr_scheduler

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break


def validate(args, model, criterion, valid_dataset, epoch):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=42))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    # Iterate over the validation set
    for i, sample in enumerate(valid_loader):
        if len(sample) == 0:
            continue
        with torch.no_grad():
            # Compute loss
            output, attn_scores = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))

    return perplexity


if __name__ == '__main__':
    args = get_args()
    args.device_id = 0

    # Set up logging to file
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    main(args)
