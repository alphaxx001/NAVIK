import json
import numpy as np
from scipy.spatial.distance import cdist

def angle_diff(a1, a2):
    diff = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi
    return diff

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Returns minimum distance and projection point."""
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2), x1, y1
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2), proj_x, proj_y

class HMMMapMatcher:
    def __init__(self, graph_file, search_radius=50.0, sigma_d=10.0, sigma_h=0.5, sigma_t=20.0):
        with open(graph_file, 'r') as f:
            data = json.load(f)
        self.segments = data['segments']
        self.origin = data['origin']
        
        self.search_radius = search_radius
        self.sigma_d = sigma_d
        self.sigma_h = sigma_h
        self.sigma_t = sigma_t
        
    def get_candidates(self, px, py, ph):
        candidates = []
        for s in self.segments:
            # Fast BBox check
            min_x = min(s['x1'], s['x2']) - self.search_radius
            max_x = max(s['x1'], s['x2']) + self.search_radius
            if px < min_x or px > max_x: continue
            
            min_y = min(s['y1'], s['y2']) - self.search_radius
            max_y = max(s['y1'], s['y2']) + self.search_radius
            if py < min_y or py > max_y: continue
            
            d, proj_x, proj_y = point_to_segment_dist(px, py, s['x1'], s['y1'], s['x2'], s['y2'])
            if d < self.search_radius:
                h_diff = abs(angle_diff(ph, s['heading']))
                if not s['oneway']:
                    h_diff_rev = abs(angle_diff(ph, (s['heading'] + np.pi) % (2*np.pi)))
                    h_diff = min(h_diff, h_diff_rev)
                
                emission = (d / self.sigma_d)**2 + (h_diff / self.sigma_h)**2
                
                candidates.append({
                    'id': s['id'],
                    'proj_x': proj_x,
                    'proj_y': proj_y,
                    'dist': d,
                    'h_diff': h_diff,
                    'emission': emission,
                    'segment': s
                })
        return candidates

    def viterbi_match(self, trajectory):
        """
        trajectory: list of dicts [{'x': x, 'y': y, 'h': h_rad}]
        Returns: list of map-matched (x, y), list of states ('MAP_MATCHED' or 'DR_FALLBACK')
        """
        if len(trajectory) == 0: return [], []
        
        # Trellis: layer_idx -> {candidate_id: {'cost': float, 'prev': id, 'proj_x': x, 'proj_y': y}}
        trellis = []
        
        # Initialization
        first_pt = trajectory[0]
        cands = self.get_candidates(first_pt['x'], first_pt['y'], first_pt['h'])
        if not cands:
            layer = {'DR_FALLBACK': {'cost': 0, 'prev': None, 'proj_x': first_pt['x'], 'proj_y': first_pt['y'], 'is_fallback': True}}
        else:
            layer = {}
            for c in cands:
                layer[c['id']] = {'cost': c['emission'], 'prev': None, 'proj_x': c['proj_x'], 'proj_y': c['proj_y'], 'is_fallback': False}
        trellis.append(layer)
        
        # Recursion
        for t in range(1, len(trajectory)):
            pt = trajectory[t]
            cands = self.get_candidates(pt['x'], pt['y'], pt['h'])
            
            prev_layer = trellis[-1]
            current_layer = {}
            
            if not cands:
                best_prev = min(prev_layer.items(), key=lambda x: x[1]['cost'])
                current_layer['DR_FALLBACK'] = {
                    'cost': best_prev[1]['cost'], 
                    'prev': best_prev[0], 
                    'proj_x': pt['x'], 
                    'proj_y': pt['y'],
                    'is_fallback': True
                }
            else:
                for c in cands:
                    best_cost = float('inf')
                    best_prev = None
                    
                    for prev_id, prev_node in prev_layer.items():
                        if prev_node['is_fallback']:
                            trans_cost = 0
                        else:
                            dp = np.sqrt((c['proj_x'] - prev_node['proj_x'])**2 + (c['proj_y'] - prev_node['proj_y'])**2)
                            d_dr = np.sqrt((pt['x'] - trajectory[t-1]['x'])**2 + (pt['y'] - trajectory[t-1]['y'])**2)
                            trans_cost = abs(dp - d_dr) / self.sigma_t
                            
                        cost = prev_node['cost'] + trans_cost + c['emission']
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = prev_id
                            
                    current_layer[c['id']] = {
                        'cost': best_cost,
                        'prev': best_prev,
                        'proj_x': c['proj_x'],
                        'proj_y': c['proj_y'],
                        'is_fallback': False
                    }
                    
            trellis.append(current_layer)
            
        # Backtracking
        matched = []
        states = []
        last_layer = trellis[-1]
        best_end = min(last_layer.items(), key=lambda x: x[1]['cost'])[0]
        
        curr_id = best_end
        for t in range(len(trajectory)-1, -1, -1):
            node = trellis[t][curr_id]
            matched.append((node['proj_x'], node['proj_y']))
            states.append("DR_FALLBACK" if node['is_fallback'] else "MAP_MATCHED")
            curr_id = node['prev']
            
        return matched[::-1], states[::-1]
