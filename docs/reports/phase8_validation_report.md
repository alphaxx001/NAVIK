# Phase 8: Robust Open-Loop Map Matching & Efficient Candidate Retrieval

## 1. Baseline
The baseline established in Phase 7C was an open-loop HMM with a 100m candidate search radius. Over the validation set, this approach yields a median drift of 44.3% but suffers a catastrophic worst-case drift of 1226% (session Vtb1), driven by unbounded DR accumulation forcing permanent fallback.

## 2 & 3. Spatial-Index Design & Equivalence
To eliminate the severe CPU bottleneck, we implemented a `cKDTree` spatial index. Road segments were discretized into structural nodes every ~20m. During search, the tree efficiently queries bounding nodes, collapses them into unique segment identities, and performs the true geometric point-to-segment analytical projection.
- **Speedup**: 1013x (0.07s vs 75.04s for 1000 queries)
- **Equivalence**: 100% exact numerical match with brute-force candidate outputs.

## 4 & 5. Candidate-Radius & Adaptive Experiments
We evaluated extending the search radius blindly (200m, 300m) and adaptively (expanding 15m/s while in fallback up to 500m). 
- **Results**: Extending the radius failed to improve median drift (fluctuating trivially between 43.9% and 44.6%) and completely failed to rescue the worst-case session (locked at 1226% drift).

## 6 & 10. HMM Robustness & False-Candidate Safety Test
The reason extending the radius fails to rescue drifted trajectories is exactly why the open-loop HMM is mathematically robust. The HMM computes transition costs based on physical distance traveled ($d_{dr}$) versus map distance jumped ($dp$). 
We constructed a deterministic **False-Candidate Safety Test** where a valid straight trajectory was suddenly subjected to an 80m parallel false candidate. 
- **Result: PASS**. The HMM robustly rejected the false candidate, preferring `DR_FALLBACK` (which costs 0 to enter) over incurring the massive emission penalty and spatial teleportation transition penalty of jumping to a distant road. 

## 7 & 9. Parallel-Road & Fallback Analysis
The HMM correctly refuses to snap to distant parallel roads or teleport across empty space. Consequently, when the ESKF drift organically exceeds the physical transition tolerance (approx 50-100m), the HMM transitions into `DR_FALLBACK`. Because the underlying ESKF trajectory never corrects its heading/position internally, it continues to diverge, permanently trapping the HMM in fallback regardless of how wide the search radius becomes.

## 11. Validation Results
- **Fixed 100m**: Median 44.3%, Worst 1226%
- **Fixed 200m**: Median 44.6%, Worst 1226%
- **Fixed 300m**: Median 43.9%, Worst 1226%
- **Adaptive**: Median 44.3%, Worst 1226%

## 12 & 13. Selected Configuration
**Strategy**: KD-Tree Indexed Fixed 100m Radius.
**Reason**: Wider or adaptive radii provide no navigational benefit because the HMM correctly penalizes and rejects distant candidates. The 100m radius is sufficient to catch standard lane/GPS drift, while the KD-tree provides a 1000x speedup for production runtime.

## 14. Known Limitations
Open-loop map matching cannot rescue a trajectory once structural divergence occurs. Closed-loop greedy matching (Phase 7D) is too brittle. We have hit the fundamental limit of post-processing decoupled architectures.
