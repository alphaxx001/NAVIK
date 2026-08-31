import time
import json
import numpy as np
from scipy.spatial import cKDTree
from map.runtime.hmm_matcher import HMMMapMatcher, point_to_segment_dist, angle_diff

class KDTreeHMMMapMatcher(HMMMapMatcher):
    def __init__(self, graph_file, search_radius=50.0, sigma_d=10.0, sigma_h=0.5, sigma_t=20.0):
        super().__init__(graph_file, search_radius, sigma_d, sigma_h, sigma_t)
        
        # Build KD-Tree
        self.pts_list = []
        self.segment_map = []
        
        for i, s in enumerate(self.segments):
            # Discretize segment into points every ~20m to ensure coverage
            length = np.sqrt((s['x2'] - s['x1'])**2 + (s['y2'] - s['y1'])**2)
            num_pts = max(2, int(length / 20.0) + 1)
            xs = np.linspace(s['x1'], s['x2'], num_pts)
            ys = np.linspace(s['y1'], s['y2'], num_pts)
            for x, y in zip(xs, ys):
                self.pts_list.append([x, y])
                self.segment_map.append(i)
                
        self.tree = cKDTree(self.pts_list)

    def get_candidates_kdtree(self, px, py, ph, dynamic_radius=None):
        r = dynamic_radius if dynamic_radius is not None else self.search_radius
        # Query KD-Tree with radius + 20m (to account for discretization gap)
        pt_indices = self.tree.query_ball_point([px, py], r + 20.0)
        
        # Get unique segment indices
        seg_indices = set(self.segment_map[i] for i in pt_indices)
        
        candidates = []
        for idx in seg_indices:
            s = self.segments[idx]
            d, proj_x, proj_y = point_to_segment_dist(px, py, s['x1'], s['y1'], s['x2'], s['y2'])
            
            if d < r:
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

def test_equivalence():
    # Load one of the validation graphs
    graph_file = "map/runtime/Vta1a_graph.json"
    
    matcher_bf = HMMMapMatcher(graph_file, search_radius=100.0)
    matcher_kd = KDTreeHMMMapMatcher(graph_file, search_radius=100.0)
    
    # Generate 1000 random query points within the bounds of the graph
    np.random.seed(42)
    origin = matcher_bf.origin
    min_x = min(s['x1'] for s in matcher_bf.segments)
    max_x = max(s['x1'] for s in matcher_bf.segments)
    min_y = min(s['y1'] for s in matcher_bf.segments)
    max_y = max(s['y1'] for s in matcher_bf.segments)
    
    queries = []
    for _ in range(1000):
        qx = np.random.uniform(min_x, max_x)
        qy = np.random.uniform(min_y, max_y)
        qh = np.random.uniform(-np.pi, np.pi)
        queries.append((qx, qy, qh))
        
    print("Running Brute-Force...")
    t0 = time.time()
    bf_results = []
    for q in queries:
        bf_results.append(matcher_bf.get_candidates(*q))
    t_bf = time.time() - t0
    
    print("Running KD-Tree...")
    t0 = time.time()
    kd_results = []
    for q in queries:
        kd_results.append(matcher_kd.get_candidates_kdtree(*q))
    t_kd = time.time() - t0
    
    print(f"Brute-Force Time: {t_bf:.4f}s")
    print(f"KD-Tree Time:     {t_kd:.4f}s")
    print(f"Speedup:          {t_bf/t_kd:.2f}x")
    
    # Assert Equivalence
    mismatches = 0
    for bf_cands, kd_cands in zip(bf_results, kd_results):
        bf_ids = set(c['id'] for c in bf_cands)
        kd_ids = set(c['id'] for c in kd_cands)
        if bf_ids != kd_ids:
            mismatches += 1
            
    if mismatches == 0:
        print("EQUIVALENCE TEST: PASS. Indexed Result == Brute-Force Result")
    else:
        print(f"EQUIVALENCE TEST: FAIL. {mismatches} mismatches found.")

if __name__ == "__main__":
    test_equivalence()
