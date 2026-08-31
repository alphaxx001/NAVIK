# Final Claims Audit for SIH Presentation

This document categorizes intended technical claims against the empirical results generated during the rigorously frozen 12-session test evaluation and prior research phases.

## 1. Deep Learning Architecture
**Claim:** "Deep learning models entirely replace classical dead-reckoning filters for navigation."
**Status:** NOT SUPPORTED. 
**Correction:** The models (SpeedNet/HeadingNet) act as *pseudo-measurements* providing kinematic bounds to a classical continuous Error-State Kalman Filter (ESKF), which remains the mathematical core of the engine.

## 2. Spatial Mapping Efficiency
**Claim:** "KD-Tree spatial indexing allows the system to process massive OSM graphs efficiently."
**Status:** SUPPORTED BY EXPERIMENT.
**Evidence:** Brute-force map queries required ~400 seconds; KD-Tree implementation resolved queries in ~0.4 seconds (a $>1000\times$ speedup), making near-real-time map-matching feasible on edge devices.

## 3. Map-Matched Navigational Improvement
**Claim:** "Snapping the trajectory to the OSM road network fixes inertial drift during blackouts."
**Status:** NOT SUPPORTED (in cases of extreme drift). SUPPORTED BY EXPERIMENT (for minor local corrections).
**Evidence:** The final test evaluation proved that once the base ESKF drifts heavily (e.g., $>100\%$ median drift), the map matcher cannot rescue the trajectory. In fact, attempting to forcefully close the loop induces catastrophic topological lock-in (Phase 11 Vtb1 explosion to 1226% error).

## 4. Continuous Operation During Blackouts
**Claim:** "The system flawlessly maintains lane-level accuracy during 30+ minute GNSS blackouts."
**Status:** FUTURE WORK.
**Evidence:** While the pipeline functions technically without crashing, smartphone-grade MEMS IMUs accumulate massive unbounded heading drift over prolonged durations. Lane-level accuracy is only achievable for short outages (1-3 minutes). Extended blackouts currently yield $~100\%$ drift errors without secondary absolute referencing.

## 5. Algorithmic Graceful Degradation
**Claim:** "The system is confidence-aware and safely defaults to dead-reckoning when map data is ambiguous or the vehicle travels off-road."
**Status:** SUPPORTED BY CODE.
**Evidence:** The open-loop decoupled architecture specifically tracks distance to the nearest valid map candidate. If this exceeds 100m, the HMM correctly emits `DR_FALLBACK` states, ensuring the user display is not forcefully snapped to an impossible geometric reality.
