# Phase 7B: Real OSM Map Validation Report

## A. Real OSM Reference Evaluation

| Session | Duration (s) | Pts | Raw RMSE | Raw Drift | MM RMSE | MM Drift | Abs Imp | Pct Imp |
|---|---|---|---|---|---|---|---|---|
| Vta10 | 148.1 | 1482 | 201.4 | 9.2% | 195.8 | 9.0% | 0.3% | 2.8% |
| Vta11 | 48.9 | 490 | 252.6 | 45.2% | 245.6 | 44.0% | 1.3% | 2.8% |
| Vta2 | 1097.0 | 10971 | 19423.9 | 127.7% | 19423.6 | 127.7% | 0.0% | 0.0% |
| Vta20 | 320.2 | 3203 | 445.3 | 27.6% | 418.6 | 25.9% | 1.7% | 6.0% |
| Vta29 | 2367.8 | 23685 | 60273.4 | 213.9% | 60273.8 | 213.9% | -0.0% | -0.0% |
| Vta3 | 62.4 | 625 | 165.6 | 38.6% | 161.4 | 37.6% | 1.0% | 2.6% |
| Vtb11 | 34.0 | 341 | 206.4 | 44.9% | 199.9 | 43.5% | 1.4% | 3.2% |
| Vtb12 | 42.6 | 427 | 155.0 | 31.3% | 137.4 | 27.8% | 3.6% | 11.4% |
| Vw10 | 63.1 | 632 | 329.1 | 36.8% | 320.8 | 35.9% | 0.9% | 2.5% |
| Vw11 | 488.8 | 4889 | 4784.3 | 83.0% | 4782.7 | 83.0% | 0.0% | 0.0% |
| Vw3 | 384.0 | 3841 | 2467.0 | 51.7% | 2466.1 | 51.7% | 0.0% | 0.0% |
| Vw5 | 99.1 | 992 | 415.4 | 37.2% | 414.0 | 37.0% | 0.1% | 0.3% |

## B. Fallback Behavior

| Session | Map-Matched % | Fallback % | Transitions | Min Dist | Mean Dist | Max Dist |
|---|---|---|---|---|---|---|
| Vta10 | 97.3% | 2.7% | 4 | 0.7m | 72.7m | 100.0m |
| Vta11 | 71.4% | 28.6% | 1 | 0.4m | 64.0m | 100.0m |
| Vta2 | 17.5% | 82.5% | 57 | 0.1m | 64.4m | 100.0m |
| Vta20 | 80.3% | 19.7% | 4 | 0.2m | 58.7m | 100.0m |
| Vta29 | 7.7% | 92.3% | 43 | 0.1m | 51.6m | 100.0m |
| Vta3 | 100.0% | 0.0% | 0 | 0.7m | 60.9m | 100.0m |
| Vtb11 | 44.3% | 55.7% | 2 | 4.0m | 69.0m | 99.4m |
| Vtb12 | 67.9% | 32.1% | 5 | 0.0m | 60.7m | 100.0m |
| Vw10 | 100.0% | 0.0% | 0 | 0.2m | 63.1m | 99.9m |
| Vw11 | 53.6% | 46.4% | 5 | 0.3m | 62.3m | 100.0m |
| Vw3 | 52.6% | 47.4% | 11 | 0.2m | 66.7m | 100.0m |
| Vw5 | 100.0% | 0.0% | 0 | 0.2m | 65.4m | 100.0m |

## C. Failure Cases

A failure case is defined as significant DR_FALLBACK (>20%) indicating the trajectory left the 100m candidate search radius.
- **Vta11**: 28.6% fallback. Max candidate dist: 100.0m.
- **Vta2**: 82.5% fallback. Max candidate dist: 100.0m.
- **Vta29**: 92.3% fallback. Max candidate dist: 100.0m.
- **Vtb11**: 55.7% fallback. Max candidate dist: 99.4m.
- **Vtb12**: 32.1% fallback. Max candidate dist: 100.0m.
- **Vw11**: 46.4% fallback. Max candidate dist: 100.0m.
- **Vw3**: 47.4% fallback. Max candidate dist: 100.0m.

## D. Aggregate Results

- **Mean Raw DR Drift**: 62.3%
- **Median Raw DR Drift**: 41.7%
- **Mean Map-Matched Drift**: 61.4%
- **Median Map-Matched Drift**: 40.5%
- **Mean Improvement**: 0.9%
- **Median Improvement**: 0.6%
- **Mean Fallback %**: 33.9%
- **Median Fallback %**: 30.3%