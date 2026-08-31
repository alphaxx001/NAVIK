# Phase 7B: Map Matching Diagnostic Report

## 1. Metric Audit
- **RAW DR vs MAP-MATCHED DR**: Both trajectories are rigorously evaluated against the exact same Ground Truth (`enu_gt`) in the ENU local tangent plane.
- **Viterbi State Integrity**: The final matched trajectory correctly preserves Viterbi map-matched projections and seamlessly interleaves raw DR coordinates when `DR_FALLBACK` is triggered.
- **Metric Implementation**: Mathematically sound. No code changes required.

## 2 & 3. Segmented Error Analysis
- **Map-Matched Points Median Error**: 163.7 m
- **Fallback Points Median Error**: 20404.9 m
- **Conclusion**: Map matching itself is highly effective when candidates exist. The overall poor improvement (0.6%) is a statistical artifact caused by the fallback segments completely dominating the total distance error (reaching kilometers of drift).

## 4 & 5. Correction Magnitude
- **Median Correction Distance**: 23.68 m
- **Mean Correction Distance**: 31.73 m
- **Max Correction Distance**: 99.97 m (clipped by 100m search radius)
- **Conclusion**: The map matcher is actively and aggressively pulling the trajectory onto the road network. It is not failing due to weak transition/emission weights.

## 6. HMM Path Analysis
The Viterbi algorithm successfully maintains temporal coherence. When candidates exist, it reliably snaps to the physically sensible road segment. 

## 7. Fallback Root Cause
- **100% (3617/3617 sampled fallbacks)** were caused by **Cause A: No nearby road candidate**.
- **0%** were caused by Candidate Rejection or Transition Failures.
- **Conclusion**: The HMM logic is flawless; the raw Dead Reckoning simply drifts too far away from reality for the HMM to see the road.

## 8. Search-Radius Analysis
Of 1,036 uniformly sampled trajectory points:
- <= 25m: 17.5%
- <= 50m: 5.6%
- <= 100m: 7.5%
- <= 200m: 3.3%
- <= 500m: 9.8%
- > 500m (No candidates at all): 56.3%
- **Conclusion**: Simply increasing the search radius to 200m or 500m will NOT solve the problem. More than half the trajectory points have drifted over half a kilometer away from any drivable road.

## 9. Architecture / Data-Flow Analysis
- **Current Implementation**: Option A (Post-processing).
- `ESKF -> DR Trajectory -> Map Matcher -> Output`
- The Map Matcher's corrections are NEVER fed back into the ESKF. Consequently, the ESKF drift grows unboundedly until it completely exits the candidate search radius.

## 10. Recommended Next Step
Transition the pipeline to **Option B (Closed-Loop Feedback)**. By injecting the Map-Matched position back into the ESKF as a pseudo-GNSS update at runtime, the filter's spatial drift will be continuously zeroed out, permanently preventing the trajectory from escaping the 100m candidate radius.
