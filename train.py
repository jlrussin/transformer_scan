# Training script for experiments with SCAN dataset

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

from data import build_scan
from models.transformer import *
from test import test
from utils import *


def train(args):

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data
    SRC, TRG, train_data, dev_data, test_data = build_scan(args.split,
                                                           args.batch_size,
                                                           device)

    # vocab
    src_vocab_size = len(SRC.vocab.stoi)
    trg_vocab_size = len(TRG.vocab.stoi)
    pad_idx = SRC.vocab[SRC.pad_token]
    assert TRG.vocab[TRG.pad_token] == pad_idx

    # Model
    if args.model_type == 'transformer':
        model = Transformer(src_vocab_size,trg_vocab_size,args.d_model,
                            args.nhead,args.num_encoder_layers,
                            args.num_decoder_layers,args.dim_feedforward,
                            args.dropout,pad_idx)
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model = model.to(device)
    model.train()

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Setup things to record
    loss_data = [] # records losses
    train_acc = [] # records train accuracy
    val_acc = [] # records validation accuracy
    test_acc = [] # records test accuracy
    best_val_acc = float('-inf') # best val accuracy (for doing early stopping)

    # Training loop:
    for epoch in range(args.num_epochs):
        for iter,batch in enumerate(train_data):
            optimizer.zero_grad()
            out = model(batch.src,batch.trg)
            loss = loss_fn(out.view(-1,trg_vocab_size),batch.trg.view(-1))
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
            train_acc = test(train_data, model, pad_idx, device, args)
            print("Training accuracy is ", train_acc)
            train_accs.append(train_acc)

            # Checkpoint on validation data
            print("Checking validation accuracy...")
            val_acc = test(val_data, model, pad_idx, device, args)
            print("Validation accuracy is ", val_acc)
            val_accs.append(val_acc)

            # Checkpoint on test data
            print("Checking test accuracy...")
            test_acc = test(test_data, model, pad_idx, device, args)
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