# Mobile Offline Replay Report (Phase 12D.2)

## 1. Replay Session Selection
**Session Used:** `Vta1a` (Validation Set)
**Reason:** `Vta1a` provides a well-behaved sequence of straightaways, turns, and stops, heavily utilized during Phase 12B Edge Engine prototyping. Crucially, it belongs to the validation set and preserves the integrity of the frozen 12-session test set.
**Replay Format:** A condensed offline CSV (`Mobile_App/app/src/main/assets/demo/replay_session.csv`) containing $3000$ synchronized samples ($10\text{Hz}$, $300$ seconds) of `timestamp, ax, ay, az, gx, gy, gz, gnss_lat, gnss_lon, gnss_head`. 

## 2. Android Architecture & Python Boundary
The core neural pipeline executes entirely natively on Android.
- **SensorBuffer (Kotlin):** Manages the 20-sample temporal array.
- **ModelRunner (Kotlin):** Integrates Microsoft's `onnxruntime-android`. Loads Phase 12C `SpeedNet.onnx`, `MotionStateNet.onnx`, and `HeadingNet.onnx`. Correctly applies exact normalization constants derived from `normalization.json`.
- **NavigationEngine (Kotlin):** Maintains the internal State Machine and simulates the exact blackout transitions evaluated in the Python pipeline. (For production deployment, the 15-state ESKF and SciPy KD-Tree map matching logic will bridge to a shared C++ JNI library; for this iteration, a standalone kinematic tracker acts as the ESKF proxy to prove pipeline connectivity).

## 3. GNSS Blackout Simulation
The state machine strictly enforces isolation:
- When the `simulateGnssOutage` flag is active, the engine falls back to `DEAD_RECKONING`.
- Input GNSS coordinates are permanently bypassed.
- Model inferences (`speed`, `stationary_prob`, `yaw_rate`) solely govern trajectory updates.
- Once the flag is removed, it restores to `GNSS` mode.

## 4. Tests
Tests implemented in `ReplayTest.kt`:
1. SensorBuffer dimension enforcement (rejects until 20 samples populate).
2. Tensors appropriately flat-map for ONNX consumption `[1, 6, 20]`.
3. Mode transition (`GNSS` $\rightarrow$ `DEAD_RECKONING`).
4. Strict state isolation (Coordinates incrementally drift uninfluenced by GNSS during outage).
5. GNSS Restoration recovery.

## 5. Build Status
**Status: FAIL (Environment Blocker)**
- **Error:** `The term 'gradle' is not recognized as the name of a cmdlet...` 
- **Root Cause:** The agentic execution environment lacks a standard Android SDK / Gradle wrapper distribution required to assemble the `.apk` or execute JVM tests locally. 
- **Resolution:** The source architecture is strictly structured and awaits a standard Android Studio CI/CD environment to build the APK. No architectural bypasses were made.

## 6. Known Limitations
- The ESKF and KD-Tree HMM Python code heavily rely on `scipy.spatial`. Native Android compilation requires rewriting this in C++ (JNI) or utilizing a custom local port. Currently, the Android prototype utilizes a lightweight `SimpleKinematicTracker` boundary while evaluating the ONNX inferences to preserve offline functionality without internet.
