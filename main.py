# Main script for gathering args, running train

import argparse
from train import train

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--split',
                    choices = ['simple','addjump'],
                    help='SCAN split to use for training and testing')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epochs')

# Models
parser.add_argument('--model_type', choices=['transformer'],
                    default='transformer', help='Type of seq2seq model to use.')
parser.add_argument('--d_model', type=int, default=12,
                    help='Dimension of inputs/outputs in transformer')
parser.add_argument('--nhead', type=int, default=2,
                    help='Number of heads in transformer multihead attention')
parser.add_argument('--num_encoder_layers', type=int, default=2,
                    help='Number of layers in transformer encoder')
parser.add_argument('--num_decoder_layers', type=int, default=2,
                    help='Number of layers in transformer decoder')
parser.add_argument('--dim_feedforward', type=int, default=20,
                    help='Dimension of feedforward layers in transformer')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')

# Output options
parser.add_argument('--results_dir', default='transformer',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='train_results.json',
                    help='Name of output data file with training loss data')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

def main(args):
    train(args)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
