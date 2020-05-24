# Training script for experiments with SCAN dataset

import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ScanDataset, get_dataset, SCAN_collate
from models.transformer import *
from test import test
from utils import *


def train(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Vocab
    with open(args.load_vocab_json,'r') as f:
        vocab = json.load(f)
    in_vocab_size = len(vocab['in_token_to_idx'])
    out_vocab_size = len(vocab['out_idx_to_token'])

    # Datasets
    dataset = get_dataset(args.split,'train',vocab)
    split_id = int(0.8*len(all_train_data))
    train_data = [all_train_data[i] for i in range(split_id)]
    val_data = [all_train_data[i] for i in range(split_id,len(all_train_data))]
    test_data = get_dataset(args.split,'test',vocab)

    # Dataloaders
    train_loader = DataLoader(train_data,args.batch_size,
                              shuffle=True,collate_fn=SCAN_collate)
    val_loader = DataLoader(val_data,args.batch_size,
                            shuffle=True,collate_fn=SCAN_collate)
    test_loader = DataLoader(test_data,args.batch_size,
                             shuffle=True,collate_fn=SCAN_collate)

    # Model
    if args.model_type == 'transformer':
        model = Transformer()
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)

    # Loss function
    loss_fn = nn.NLLLoss(reduction='mean',ignore_index=-100)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Setup things to record
    loss_data = [] # records losses
    train_acc = [] # records train accuracy
    val_acc = [] # records validation accuracy
    test_acc = [] # records test accuracy
    best_val_acc = -np.inf # best val accuracy (for doing early stopping)

    # Training loop:
    for epoch in range(num_epochs):
        for iter,sample in enumerate(train_loader):
            # Forward pass
            instructions, true_actions, _, _ = sample
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            optimizer.zero_grad()
            actions,padded_true_actions = model(instructions,true_actions)
            # Compute NLLLoss
            true_actions = padded_true_actions.to(device)
            loss = loss_fn(actions,padded_true_actions)
            # Backward pass
            loss.backward()
            optimizer.step()
            # Record loss
            if iter % args.record_loss_every == 0:
                loss_datapoint = loss.data.item()
                print('Epoch:', epoch,
                      'Iter:', iter,
                      'Loss:', loss_datapoint)
                loss_data.append(loss_datapoint)

        # Checkpoint
        if epoch_count % args.checkpoint_every == 0:
            # Checkpoint on train data
            print("Checking training accuracy...")
            train_acc = test(train_loader, model, device, args)
            print("Training accuracy is ", train_acc)
            train_accs.append(train_acc)

            # Checkpoint on validation data
            print("Checking validation accuracy...")
            val_acc = test(val_loader, model, device, args)
            print("Validation accuracy is ", val_acc)
            val_accs.append(val_acc)

            # Checkpoint on test data
            print("Checking test accuracy...")
            test_acc = test(test_loader, model, device, args)
            print("Test accuracy is ", test_acc)
            test_accs.append(test_acc)

            # Write stats file
            results_path = 'results/%s' % (args.results_dir)
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            stats = {'loss_data':loss_data,
                     'train_accs':train_accs,
                     'val_accs':val_accs,
                     'test_accs':test_accs}
            results_file_name = '%s/%s' % (results_path,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)

            # Save model weights
            if val_acc > best_val_acc: # use val to decide to save
                best_val_acc = val_acc
                if args.checkpoint_path is not None:
                    torch.save(model.state_dict(),
                               args.checkpoint_path)
