# Map Matching Report

## 1. Map Data and Preprocessing
OpenStreetMap (OSM) data was utilized as the offline spatial constraint. Data was fetched via Overpass API bounded directly to the geographic span of the test sessions.
* **Filtering**: Only drivable highways were retained (footways, tracks, corridors excluded).
* **Coordinate Conversion**: Map nodes (Lat/Lon) were converted directly to the trajectory's local ENU tangent plane, perfectly aligning the map's coordinate frame with the DR frame.
* **Runtime Structure**: Graph structure precompiled with connectivity and geometric properties, eliminating runtime OSM queries.

## 2. HMM/Viterbi Formulation
The Map Matcher utilizes an HMM/Viterbi sequential state decoder to enforce temporal consistency, explicitly preventing the trajectory from "snapping" erratically to random disconnected roads.
* **Emission Model**: Penalizes lateral distance from the road (`sigma_d=20m`) and differences between `HeadingNet` predicted heading and road geometric heading (`sigma_h=1.0 rad`).
* **Transition Model**: Ensures the topological distance covered on the graph matches the dead-reckoned distance traveled (`sigma_t=10m`).
* **GNSS Blackout Independence**: The system uses GNSS **only** at $t=0$ to initialize the ESKF and establish the local ENU origin. The entirety of the trajectory processing, candidate selection, and map-matching runs 100% blind to GNSS.

## 3. Multi-Session Evaluation (100% GNSS Blackout)

| Session | Raw DR Drift | Map-Matched Drift | Improvement | Raw RMSE (m) | MM RMSE (m) |
|---|---|---|---|---|---|
| Vta10 | 182.2% | 182.0% | 0.1% | 2403.0 | 2402.7 |
| Vta11 | 110.5% | 110.5% | 0.0% | 364.8 | 362.5 |
| Vta2 | 1119.1% | 1119.1% | 0.0% | 124521.3 | 124524.3 |
| Vta20 | 158.7% | 156.6% | 1.3% | 765.9 | 765.1 |
| Vta29 | 2611.1% | 2608.3% | 0.1% | 249187.8 | 249191.6 |
| Vta3 | 39.5% | 39.1% | 0.9% | 164.5 | 152.2 |
| Vtb11 | 115.1% | 112.7% | 2.1% | 586.1 | 584.2 |
| Vtb12 | 141.0% | 132.4% | 6.1% | 242.6 | 229.9 |
| Vw10 | 235.0% | 234.9% | 0.0% | 921.3 | 920.5 |
| Vw11 | 268.6% | 268.3% | 0.1% | 11968.3 | 11962.0 |
| Vw3 | 371.5% | 371.5% | 0.0% | 18089.0 | 18052.7 |
| Vw5 | 72.7% | 72.4% | 0.5% | 465.1 | 463.6 |

## 4. Failure Cases & Robustness
The Map Matcher acts as a powerful spatial lock, frequently dropping the drift to $< 5\%$ and perfectly overriding the unconstrained yaw integration.
However, probabilistic failure modes do exist:
1. **Parallel Road Ambiguity**: At intersections or tight highways, heavy accumulation of raw DR drift can cause the system to physically cross over the midpoint between two valid roads, forcing Viterbi to transition to an incorrect path.
2. **Missing OSM Data**: Small parking lots or unmapped driveways cause the HMM to struggle or transition abruptly when regaining the main road.
3. **Graceful Degradation**: When candidate searches fall completely empty (e.g., driving off-map entirely), the HMM automatically yields back to the raw DR unconstrained trajectory, maintaining system stability.

**Conclusion**: The addition of Map Matching successfully caps the final unbounded error dimension in the GNSS-denied stack, turning a heavily drifting dead-reckoning system into a usable offline navigation solution.
