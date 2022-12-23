import os
import torch

from utils import * 
from model import _models

def predict():
    model = _models("densenet161", num_classes=2, drop_rate=0)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join("weights", "densenet161-10x", "last.pt")))
    testloader = get_testloader("images/test/", "images/label/test.csv", batch_size=1)
    total_probs = None
    total_preds = None
    true_labels = None
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels = labels if true_labels is None else torch.cat((true_labels, labels),dim=0)
            total_probs = outputs if total_probs is None else torch.cat((total_probs, outputs),dim=0)
            total_preds = preds if total_preds is None else torch.cat((total_preds, preds),dim=0)

    return total_probs.cpu(), total_preds.cpu(), true_labels.cpu().numpy()
    
if __name__ == "__main__":
    predict()