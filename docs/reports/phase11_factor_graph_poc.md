# Phase 11: Tightly Coupled Factor Graph / MAP Optimization POC

## 1. Motivation
The objective was to determine whether a fixed-lag trajectory smoothing architecture could overcome the two fundamental limitations established in prior phases: the unrecoverable drift of open-loop HMM tracking (Phase 8), and the catastrophic topological lock-in of greedy closed-loop and Particle Filters (Phases 7D & 10).

## 2-8. Formulation & Design
- **State Representation**: $X_k = [p_x, p_y, v_x, v_y, \theta]$. The trajectory was downsampled to 1Hz.
- **Temporal Window**: Fixed-lag optimization over 30 seconds (30 states).
- **Factors**: 
  - Kinematic integration (position, velocity).
  - Neural network constraints (SpeedNet, HeadingNet via the ESKF baseline).
  - Map constraint: KD-tree geometric projection to the nearest plausible road segment.
- **Robust Loss**: A `soft_l1` (Huber-like) loss was applied to the map constraints to prevent temporary false-candidate snapping from warping the entire window geometry.
- **Optimization**: Non-linear least squares (Levenberg-Marquardt).

## 9 & 11. Validation & Ablation Results
- **No Map (IMU + Neural Nets Only)**: Median Drift 26.3% | Worst Drift 121.6%
- **Robust Map Factor Graph**: Median Drift 34.1% | Worst Drift 1226.5%
- **False-Candidate Safety Test**: PASS (The robust loss and window successfully ignored a temporary 80m false candidate in a clean topology).

## 10 & 13. Vtb1 Analysis & Failure Modes
The factor graph **catastrophically failed** on Vtb1. While the pure unmapped inertial trajectory (No Map) had a drift of 121.6%, introducing the Map factors actively warped the trajectory off a cliff, dragging the drift to 1226.5%. 

**Root Cause**: Gradient-based optimizers (like LM or Gauss-Newton) operate on continuous, differentiable manifolds. However, road networks are fundamentally discrete, integer topological graphs. When the vehicle is physically located between two parallel roads, the gradients will blindly pull the trajectory into the nearest local geometric minimum (the wrong road). Once pulled into the wrong local minimum, the 30-second temporal window becomes a liability—it rigidly locks the trajectory into the wrong shape, actively fighting the true inertial physics.

## 12. Computational Cost
The computational burden in Python was astronomical:
- **Vta1a**: 3348 seconds (~55 minutes)
- **Vtb1**: 3362 seconds (~56 minutes)
This severely violates the computational budget for a real-time or near-real-time vehicular system.

## 14 & 15. Limitations & Recommendation
Continuous MAP optimization cannot natively solve discrete topological ambiguity. 
**Recommendation**: 
We have definitively exhausted standard closed-loop feedback paradigms (Greedy, Particle Filter, and Continuous Factor Graphs). The underlying truth is that map matching relies on integer topologies, while the ESKF relies on continuous physics. We must permanently abandon attempts to inject map geometry directly into the ESKF state. The architecture should revert to a decoupled **Open-Loop HMM (Phase 8)**, but we must fix the `DR_FALLBACK` divergence by improving the fundamental accuracy of the baseline ESKF (e.g., via Zero Velocity Updates, Neural IMU calibration, or stronger kinematic constraints) rather than relying on the map to magically fix dead-reckoning.
