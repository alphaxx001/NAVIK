import json
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.data.dataset_loader import IOVNBDProductionDataset

def tune_motion_state():
    device = torch.device("cpu")
    m_cfg = json.load(open("configs/motion_state_training.json"))
    
    val_dataset = IOVNBDProductionDataset(split="val", stride_override=1)
    
    mm = m_cfg['model']
    motionnet = MotionStateNet(mm['in_channels'], mm['cnn_channels'], mm['kernel_size'], mm['gru_hidden'], mm['gru_layers'])
    motionnet.load_state_dict(torch.load(m_cfg['paths']['best_model'], map_location=device, weights_only=True))
    motionnet.eval()
    
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
    preds, ys = [], []
    with torch.no_grad():
        for bx, by, _ in loader:
            p = motionnet(bx.to(device)).cpu().numpy()
            preds.append(p)
            ys.append(by.numpy())
            
    preds = np.vstack(preds).flatten()
    ys = np.vstack(ys).flatten()
    
    t_std, t_mean = val_dataset.target_std, val_dataset.target_mean
    y_real = (ys * t_std) + t_mean
    
    y_class = (y_real < m_cfg['training']['stationary_threshold']).astype(int)
    
    precisions, recalls, thresholds = precision_recall_curve(y_class, preds)
    
    # Calculate F1 for all thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # We want to maximize F1, but ensure precision is at least 0.85 (to avoid dangerous false ZUPTs)
    valid_idx = np.where(precisions[:-1] >= 0.85)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
        best_thresh = thresholds[best_idx]
    else:
        best_idx = np.argmax(f1_scores[:-1])
        best_thresh = thresholds[best_idx]
        
    print(f"Optimal Validation Threshold: {best_thresh:.4f}")
    print(f"Validation Precision: {precisions[best_idx]:.4f}")
    print(f"Validation Recall: {recalls[best_idx]:.4f}")
    print(f"Validation F1: {f1_scores[best_idx]:.4f}")
    
    # Save this threshold to config
    m_cfg['inference'] = {'optimal_threshold': float(best_thresh)}
    with open("configs/motion_state_training.json", "w") as f:
        json.dump(m_cfg, f, indent=4)
        
if __name__ == "__main__":
    tune_motion_state()
