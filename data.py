#!/usr/bin/env python
# Utilities for dealing with SCAN dataset

import torch
from torch.utils.data import Dataset

class ScanDataset(Dataset):
    def __init__(self, file, vocab=None):

        self.file = file

        # Read all data into memory
        with open(file,'r') as f:
            raw_data = f.readlines()

        # Separate data into instructions and actions
        text_instructions = []
        text_actions = []
        IN = True
        for example in raw_data:
            new_instruction = ['<SOS>']
            new_action = ['<SOS>']
            for word in example.split():
                if word == 'IN:':
                    IN = True
                    continue
                elif word == 'OUT:':
                    IN = False
                    continue
                if IN:
                    new_instruction.append(word)
                else:
                    new_action.append(word)
            new_instruction.append('<EOS>')
            new_action.append('<EOS>')
            text_instructions.append(new_instruction)
            text_actions.append(new_action)
        self.text_instructions = text_instructions
        self.text_actions = text_actions

        # Build vocabulary
        if vocab is None:
            instructions_vocab = {}
            for s in self.text_instructions:
                for token in s:
                    if token not in instructions_vocab:
                        instructions_vocab[token] = len(instructions_vocab)
            instructions_vocab['<NULL>'] = len(instructions_vocab)
            actions_vocab = {}
            for s in self.text_actions:
                for token in s:
                    if token not in actions_vocab:
                        actions_vocab[token] = len(actions_vocab)
            self.vocab = {'in_token_to_idx': instructions_vocab,
                          'out_token_to_idx': actions_vocab,
                          'in_idx_to_token': {idx:token for token,idx in instructions_vocab.items()},
                          'out_idx_to_token': {idx:token for token,idx in actions_vocab.items()}}
        else:
            self.vocab = vocab

    def __getitem__(self, index):
        # Generate instruction tensor
        in_vocab_size = len(self.vocab['in_token_to_idx'])
        text_instruction = self.text_instructions[index]
        instruction = []
        for i,token in enumerate(text_instruction):
            idx = int(self.vocab['in_token_to_idx'][token])
            idx = torch.tensor(idx)
            instruction.append(idx)
        # Generate action tensor
        out_vocab_size = len(self.vocab['out_token_to_idx'])
        text_action = self.text_actions[index]
        action = []
        for i,token in enumerate(text_action):
            idx = int(self.vocab['out_token_to_idx'][token])
            idx = torch.tensor(idx)
            action.append(idx)
        return (instruction, action, text_instruction, text_action)

    def __len__(self):
        return len(self.text_instructions)


def get_dataset(split, partition, vocab):
    if split == 'simple' and partition == 'train':
        fn = '../data/scan/simple/tasks_train_simple.txt'
    elif split == 'simple' and partition == 'test':
        fn = '../data/scan/simple/tasks_test_simple.txt'
    elif split == 'addjump' and partition == 'train':
        fn = '../data/scan/addjump/tasks_train_addprim_jump.txt'
    elif split == 'addjump' and partition == 'test':
        fn = '../data/scan/addjump/tasks_test_addprim_jump.txt'

    dataset = ScanDataset(fn,vocab)
    return dataset


def SCAN_collate(batch):
    transposed = list(zip(*batch))
    instructions = [torch.tensor(i) for i in transposed[0]]
    actions = [torch.tensor(a) for a in transposed[1]]
    instructions.sort(key=lambda l:-len(l))
    actions.sort(key=lambda l:-len(l))
    return instructions, actions, transposed[2], transposed[3]
