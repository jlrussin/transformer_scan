# Utilities for dealing with SCAN dataset

import os
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset


def build_scan(split, batch_size, device):
    # Get paths and filenames of each partition of split
    if split == 'simple':
        path = 'data/scan/simple/'
    elif split == 'addjump':
        path = 'data/scan/addjump/'
    else:
        assert split not in ['simple','addjump'], "Unknown split"
    train_path = os.path.join(path,'train')
    dev_path = os.path.join(path,'dev')
    test_path = os.path.join(path,'test')
    exts = ('.src','.trg')

    # Fields for source (SRC) and target (TRG) sequences
    SRC = Field(init_token='<sos>',eos_token='<eos>')
    TRG = Field(init_token='<sos>',eos_token='<eos>')
    fields = (SRC,TRG)

    # Build datasets
    train_ = TranslationDataset(train_path,exts,fields)
    dev_ = TranslationDataset(dev_path,exts,fields)
    test_ = TranslationDataset(test_path,exts,fields)

    # Build vocabs: fields ensure same vocab used for each partition
    SRC.build_vocab(train_)
    TRG.build_vocab(train_)

    # BucketIterator ensures similar sequence lengths to minimize padding
    train, dev, test = BucketIterator.splits((train_, dev_, test_),
        batch_size = batch_size, device = device)

    return SRC, TRG, train, dev, test
