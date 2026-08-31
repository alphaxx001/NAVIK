# Phase 10: Map-Constrained Particle Filter Validation

## 1. Motivation
The objective was to bridge the gap between robust (but unrecoverable) open-loop tracking and precise (but brittle) greedy closed-loop tracking. The Particle Filter aimed to achieve this by maintaining $N$ distinct ESKF instances, allowing map corrections to be applied independently to different topological hypotheses without corrupting a single global state.

## 2-7. Particle Architecture & Map Feedback
- **State**: $N$ independently propagated `ESKF_ClosedLoop` instances.
- **Propagation**: Standard IMU + ZUPT/NHC integration.
- **Observation**: 1Hz map queries via the Phase 8 KD-Tree.
- **Likelihood**: Gaussian weight based on spatial distance and heading error.
- **Feedback**: Particles with valid topological candidates received independent closed-loop geometric updates (bounded to 5m maximum corrections).
- **Resampling**: Standard ESS (Effective Sample Size) resampling triggered at $ESS < N/2$.

## 9 & 11. Vtb1 Analysis & Validation Results
The validation results demonstrated severe instability depending on particle count and random seed:
- **N=50**: Median Drift 72.3% | Worst Drift 132.5% (Vtb1)
- **N=100**: Median Drift 15.6% | Worst Drift 3048.0% (Vtb1)

While $N=100$ successfully reduced the median drift across the dataset, it completely collapsed during `Vtb1`. The filter encountered a geometrically attractive false road, assigned high likelihoods to particles snapping to it, and during resampling, pruned all correct hypotheses. Once the entire filter population locked onto the wrong path, the closed-loop feedback mathematically dragged the ESKFs completely off the true trajectory, resulting in an unprecedented 3048.0% drift (significantly worse than Phase 7D's 1611%).

## 12. Computational Cost
The computational burden of maintaining distinct covariance matrices is prohibitive in pure Python:
- **N=50**: 756.88 seconds
- **N=100**: 1739.20 seconds (approx 29 minutes)

## 13. Safety Tests
**Extended False-Candidate Test: PASS**. In the simplified synthetic test, the PF successfully maintained the true hypothesis against a temporary 5-timestep occlusion/spoof. However, it failed the real-world equivalent in `Vtb1`.

## 14 & 15. Limitations & Recommendation
**Limitations**:
1. **Particle Collapse**: Resampling is inherently a greedy pruning mechanism. If a false road looks better for just long enough to trigger resampling, the true hypothesis is permanently destroyed.
2. **Computational Overhead**: Propagating $N$ 15-DOF inertial filters is highly inefficient compared to a single filter.
3. **Instability**: The system's performance wildly oscillates based on particle density.

**Recommendation**:
Abandon the Particle Filter. Forward-filtering approaches (whether EKF, PF, or Greedy HMM) fundamentally struggle with map matching because they are forced to discard hypotheses *before* observing future disambiguating geometry. We must pivot to **Tightly-Coupled Factor Graph Optimization (MAP Smoothing)**, which evaluates the entire sliding window jointly, allowing future trajectory shapes to correct past topological ambiguities without relying on brittle resampling.
