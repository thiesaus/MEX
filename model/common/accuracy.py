

from tqdm import tqdm

import torch

def test_accuracy(model,dataloader):
    model.eval()
    TP, FP, FN = 0, 0, 0
    assert dataloader.batch_size == 1
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
        # for batch_idx, data in enumerate(dataloader):
            # load
       
            # forward
           
            expressions = data['target_expressions']
            expressions = expressions[0].split(',')
            labels = data['target_labels'][0]
            # forward
            inputs = dict(
                local_images=data['cropped_images'].cuda().repeat_interleave(len(expressions), dim=0),
                global_image=data['global_images'].cuda().repeat_interleave(len(expressions), dim=0),
                sentences=expressions,
            )
            logits = model(inputs)['scores'].cpu()
            # evaluate
            TP += ((logits >= 0) * (labels == 1)).sum().item()
            FP += ((logits >= 0) * (labels == 0)).sum().item()
            FN += ((logits < 0) * (labels == 1)).sum().item()
    if TP == 0:
        PRECISION = 0
        RECALL = 0
    else:
        PRECISION = TP / (TP + FP) * 100
        RECALL = TP / (TP + FN) * 100
    print(TP, FP, FN)
    return PRECISION, RECALL