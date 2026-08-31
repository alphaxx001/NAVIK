import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ml.models.speednet.speednet import SpeedNet
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.models.heading.heading_net import HeadingNet
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from ml.data.dataset_loader import IOVNBDProductionDataset
from map.runtime.hmm_matcher import HMMMapMatcher
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver
from scipy.spatial.transform import Rotation as R
from ml.evaluation.evaluate_map_matching import get_predictions, run_pipeline

def run_diagnostics():
    device = torch.device("cpu")
    s_cfg = json.load(open("configs/speednet_training.json"))
    m_cfg = json.load(open("configs/motion_state_training.json"))
    h_cfg = json.load(open("configs/heading_training.json"))
    
    test_dataset = IOVNBDProductionDataset(split="test", stride_override=1, target_type="yaw_rate")
    
    speednet = SpeedNet(s_cfg['model']['in_channels'], s_cfg['model']['cnn_channels'], s_cfg['model']['kernel_size'], s_cfg['model']['gru_hidden'], s_cfg['model']['gru_layers'])
    speednet.load_state_dict(torch.load(s_cfg['paths']['best_model'], map_location=device, weights_only=True)); speednet.eval()
    
    motionnet = MotionStateNet(m_cfg['model']['in_channels'], m_cfg['model']['cnn_channels'], m_cfg['model']['kernel_size'], m_cfg['model']['gru_hidden'], m_cfg['model']['gru_layers'])
    motionnet.load_state_dict(torch.load(m_cfg['paths']['best_model'], map_location=device, weights_only=True)); motionnet.eval()
    
    headingnet = HeadingNet(h_cfg['model']['in_channels'], h_cfg['model']['cnn_channels'], h_cfg['model']['kernel_size'], h_cfg['model']['gru_hidden'], h_cfg['model']['gru_layers'])
    headingnet.load_state_dict(torch.load(h_cfg['paths']['best_model'], map_location=device, weights_only=True)); headingnet.eval()
    
    p_speed_norm, p_motion, p_heading_norm, sess_arr = get_predictions(test_dataset, speednet, motionnet, headingnet, device)
    
    t_mean_v, t_std_v = 10.4019, 7.2721
    p_speed = (p_speed_norm * t_std_v) + t_mean_v
    t_mean_y, t_std_y = 0.0, 0.2
    p_heading = (p_heading_norm * t_std_y) + t_mean_y
    
    p_class = (p_motion > 0.569).astype(int)
    p_class_hyst = np.zeros_like(p_class)
    for i in range(2, len(p_class)):
        if p_class[i] == 1 and p_class[i-1] == 1 and p_class[i-2] == 1:
            p_class_hyst[i] = 1; p_class_hyst[i-1] = 1; p_class_hyst[i-2] = 1
            
    sessions = np.unique(sess_arr)
    
    # Track stats
    mm_errors = []
    fb_errors = []
    correction_dists = []
    fb_causes = {'A': 0, 'B/C/D': 0}
    rad_distrib = {25: 0, 50: 0, 100: 0, 200: 0, 500: 0}
    rad_pts = 0
    
    for sess in sessions:
        graph_file = f"map/runtime/{sess}_graph.json"
        if not os.path.exists(graph_file): continue
            
        idx = (sess_arr == sess)
        pos_dr, head_dr, pos_gt, v_preds, timestamps = run_pipeline(sess, p_speed[idx], p_class_hyst[idx], p_heading[idx])
        
        trajectory = []
        for i in range(0, len(pos_dr), 10):
            trajectory.append({'x': pos_dr[i,0], 'y': pos_dr[i,1], 'h': head_dr[i]})
            
        matcher = HMMMapMatcher(graph_file, search_radius=100.0, sigma_d=20.0, sigma_h=1.0, sigma_t=10.0)
        matched_pts_down, states_down = matcher.viterbi_match(trajectory)
        
        # Diagnostics
        matcher_500 = HMMMapMatcher(graph_file, search_radius=500.0)
        
        for t, pt in enumerate(trajectory):
            # Fallback causes
            if states_down[t] == 'DR_FALLBACK':
                cands = matcher.get_candidates(pt['x'], pt['y'], pt['h'])
                if len(cands) == 0:
                    fb_causes['A'] += 1
                else:
                    fb_causes['B/C/D'] += 1
            else:
                c_dist = np.linalg.norm(np.array([pt['x'], pt['y']]) - np.array(matched_pts_down[t]))
                correction_dists.append(c_dist)
                
            # Radius analysis (subsample every 5th to save time)
            if t % 5 == 0:
                cands_500 = matcher_500.get_candidates(pt['x'], pt['y'], pt['h'])
                rad_pts += 1
                if cands_500:
                    min_d = min(c['dist'] for c in cands_500)
                    if min_d <= 25: rad_distrib[25] += 1
                    elif min_d <= 50: rad_distrib[50] += 1
                    elif min_d <= 100: rad_distrib[100] += 1
                    elif min_d <= 200: rad_distrib[200] += 1
                    elif min_d <= 500: rad_distrib[500] += 1
                    
        # Error separate
        matched_arr = np.zeros_like(pos_dr[:, 0:2])
        states_arr = []
        for i in range(len(pos_dr)):
            low_idx = i // 10
            high_idx = min(low_idx + 1, len(matched_pts_down) - 1)
            alpha = (i % 10) / 10.0
            p1 = matched_pts_down[low_idx]
            p2 = matched_pts_down[high_idx]
            matched_arr[i, 0] = p1[0] * (1 - alpha) + p2[0] * alpha
            matched_arr[i, 1] = p1[1] * (1 - alpha) + p2[1] * alpha
            states_arr.append(states_down[low_idx])
            
        err_mm_all = np.linalg.norm(matched_arr - pos_gt, axis=1)
        
        for i in range(len(states_arr)):
            if states_arr[i] == 'MAP_MATCHED':
                mm_errors.append(err_mm_all[i])
            else:
                fb_errors.append(err_mm_all[i])
                
        # Zoomed Plots for Vta10, Vta29, Vw11
        if sess in ['Vta10', 'Vta29', 'Vw11']:
            plt.figure(figsize=(12, 12))
            
            # Ground truth
            plt.plot(pos_gt[:, 0], pos_gt[:, 1], 'k--', label='Ground Truth')
            plt.plot(pos_dr[:, 0], pos_dr[:, 1], 'r-', alpha=0.5, label='Raw DR')
            
            # Match
            plt.plot(matched_arr[:, 0], matched_arr[:, 1], 'b-', linewidth=2, label='Matched')
            
            fb_x = [matched_arr[i,0] for i in range(len(states_arr)) if states_arr[i] == 'DR_FALLBACK']
            fb_y = [matched_arr[i,1] for i in range(len(states_arr)) if states_arr[i] == 'DR_FALLBACK']
            if fb_x:
                plt.scatter(fb_x, fb_y, color='orange', s=10, label='DR Fallback', zorder=4)
                
            plt.legend()
            plt.title(f'Diagnostic Map Matching - {sess}')
            plt.axis('equal')
            plt.savefig(f"docs/reports/{sess}_diagnostic_plot.png")
            plt.close()

    # Save Diagnostic JSON
    diag = {
        'mm_median_err': float(np.median(mm_errors)) if mm_errors else 0.0,
        'fb_median_err': float(np.median(fb_errors)) if fb_errors else 0.0,
        'correction_dist_median': float(np.median(correction_dists)) if correction_dists else 0.0,
        'correction_dist_mean': float(np.mean(correction_dists)) if correction_dists else 0.0,
        'correction_dist_max': float(np.max(correction_dists)) if correction_dists else 0.0,
        'fb_causes': fb_causes,
        'rad_distrib': rad_distrib,
        'rad_pts': rad_pts
    }
    with open("docs/reports/diagnostic_stats.json", "w") as f:
        json.dump(diag, f, indent=4)
        
    print("Diagnostics complete. Stats written to docs/reports/diagnostic_stats.json.")

if __name__ == "__main__":
    run_diagnostics()
