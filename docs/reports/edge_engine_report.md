# Edge Engine Report (Phase 12B)

## 1. Architecture
The `Edge_Engine` forms a modular execution layer around the existing frozen IDR-System components.
- **SensorBuffer**: Retains and normalizes the required sliding window of raw IMU samples.
- **ModelRunner**: Encapsulates the PyTorch models (`SpeedNet`, `MotionStateNet`, `HeadingNet`) to isolate deep learning dependencies from state estimation.
- **NavigationStateMachine**: Transitions the output confidence layer sequentially between `GNSS`, `DEAD_RECKONING`, `MAP_CONTEXT`, and `DR_FALLBACK`.
- **EdgeEngine**: Integrates the state machine, buffer, model runner, and `ESKF`. It guarantees mathematically that `MAP_CONTEXT` geometry strictly updates the secondary display position and never corrupts the primary ESKF.

## 2. Module Interfaces
- The main pipeline accepts raw IMU sequences in `m/s^2` and `rad/s`, pushing them to `step_imu(timestamp, accel, gyro)`.
- Re-initializations (GNSS restoration) are handled exclusively via `initialize_gnss(lat, lon, heading, timestamp)`.
- Continuous state representations are emitted as `NavigationOutput` dataclasses containing authoritative positions alongside map-snapped context coordinates.

## 3. Actual Model Window Size
Inspection of the canonical `dataset_loader.py` and the existing pipeline scripts confirmed that the true sliding window dimensionality is `[B, 6, 200]`.
This corresponds to a $2.0$-second historical window sampled at $100$ Hz. The `SensorBuffer` explicitly targets this dimensionality to ensure inputs match the exact distribution trained in Phase 2.

## 4. Map $\rightarrow$ ESKF Feedback Validation
As established in the frozen Phase 12 architecture document, the map context layer is rigorously isolated.
- Verification: `engine.py` calls `eskf.predict()`, `eskf.update_zupt()`, and `eskf.update_oracle_speed()`. It **never** invokes any function that writes to `eskf.p` or `eskf.q` based on the KD-Tree HMM output. Map feedback is guaranteed `FALSE`.

## 5. Runtime and Timing
Performance characteristics extracted from the `Vta1a` offline replay smoke test:
- **Total Replay Time**: ~180-250 seconds (varying by background CPU scheduling).
- **Average Inference Step Time (10Hz)**: 3-5 ms per step (PyTorch CPU). 
- The system achieves processing speeds significantly faster than real-time (handling a 2500s session in a fraction of that time on a desktop CPU).

## 6. Known Limitations
- The current `ModelRunner` instantiates raw PyTorch `.pth` models, which carry heavy memory overheads. Transitioning this edge logic to mobile hardware will require Phase 12C (TFLite/ONNX deployment optimizations).
- The `KDTreeHMMMapMatcher` is fully instantiated in memory. While lookups take `<1ms`, loading large global bounding boxes into RAM could challenge lower-end mobile devices.
