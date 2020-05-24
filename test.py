# Function for testing models

import numpy as np
import torch

def test(dataloader, model, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = [] # list of booleans indicating whether correct
        for batch in dataloader:
            instructions, true_actions, _, _ = batch
            batch_size = len(instructions)
            out_vocab_size = model.out_vocab_size
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            actions,padded_true_actions = model(instructions, true_actions)

            # Manually unpad with mask to compute accuracy
            mask = padded_true_actions == -100
            max_actions = torch.argmax(actions,dim=1)
            correct_actions = max_actions == padded_true_actions
            correct_actions = correct_actions + mask # Add boolean mask
            correct_actions = correct_actions.cpu().numpy()
            correct_trials = np.all(correct_actions,axis=1).tolist()
            all_correct_trials = all_correct_trials + correct_trials
    accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy
