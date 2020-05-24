# Function for testing models

import numpy as np
import torch

def test(data, model, pad_idx, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = [] # list of booleans indicating whether correct
        for batch in data:
            out = model(batch.src, batch.trg)
            preds = torch.argmax(out,dim=2)
            correct_pred = preds == batch.trg
            mask = batch.trg == pad_idx # mask out padding
            correct = torch.logical_or(mask,correct_pred)
            correct = correct.all(0).tolist()
            all_correct_trials += correct

    accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy
