# Phase 9: Multi-Hypothesis HMM Validation

## 1. Motivation
We sought to determine whether maintaining multiple topological hypotheses (Top-K paths) could bridge the gap between brittle closed-loop feedback and structurally trapped open-loop HMMs, without incurring the extreme complexity of Particle Filters.

## 2. Failure mode from Phase 7D
Greedy closed-loop map matching (K=1, lag=0) proved highly brittle. When the system encountered ambiguous intersections or parallel roads, the greedy selector snapped to the wrong road segment. Because this incorrect projection was immediately fed back into the ESKF, it actively dragged the inertial filter away from reality, compounding the error irreversibly (e.g., creating 1611% drift in `Vtb1`).

## 3. Failure mode from Phase 8
Robust open-loop map matching correctly penalizes implausible topological leaps. However, because the map-matcher's output is purely post-processed and never fed back into the ESKF, the ESKF trajectory eventually drifts beyond the 100m search radius. Once out of bounds, the HMM is permanently trapped in `DR_FALLBACK`.

## 4 & 5 & 6. Multi-Hypothesis Design & Scoring
We implemented a Beam-Search (Fixed-Lag Smoothing) Multi-Hypothesis HMM. 
- **State Representation**: At time $t$, the system maintains the Top-K distinct trajectory histories (paths) spanning a sliding window. 
- **Scoring**: Each path accumulates the standard HMM emission (distance, heading) and transition (physical consistency) costs. 

## 7. Delayed Commitment
Instead of committing instantaneously, the system waits for 2 seconds (20 timesteps) of temporal evidence to accumulate across all K hypotheses before outputting the oldest state from the globally best path.

## 8 & 11. Parallel-Road Handling & Safety Tests
The multi-hypothesis delayed-commitment architecture handles ambiguity elegantly. 
**Extended False-Candidate Test: PASS**. When a highly scored false candidate was injected for 5 consecutive timesteps, the true path hypothesis survived inside the Top-K pool because it had lower transition penalties over time. The false path was ultimately pruned, and the system did not lock onto the wrong road.

## 9 & 10. Validation Results & Runtime
- **K=1**: Drift 44.3% (Worst 1226.5%) | Runtime 0.74s
- **K=3**: Drift 44.3% (Worst 1226.5%) | Runtime 1.27s
- **K=5**: Drift 44.3% (Worst 1226.5%) | Runtime 1.59s
- **K=10**: Drift 44.3% (Worst 1226.5%) | Runtime 2.93s

## 12. Limitations
The results are mathematically identical across all K values. Why? Because the multi-hypothesis tracker is open-loop. When the raw DR trajectory drifts 1 kilometer away in `Vtb1`, there are zero candidates within the 100m radius. Whether the system tracks 1 fallback hypothesis or 10 fallback hypotheses is irrelevant; the true road is physically out of bounds. Maintaining Top-K paths filters out short-term ambiguity but is powerless against long-term structural divergence.

## 13. Recommendation
We must proceed to a **Particle Filter** or tightly-coupled MAP optimizer. Open-loop tracking cannot rescue diverged states, and greedy closed-loop tracking is too brittle. We need an architecture that actually instantiates multiple ESKF states (particles) concurrently, allowing map-matching feedback to occur per-hypothesis rather than globally, thereby naturally pruning diverged filters while maintaining stable ones.
