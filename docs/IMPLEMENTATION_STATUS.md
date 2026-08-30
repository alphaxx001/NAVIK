# Implementation Status

## CURRENT

*   **`AI_Model/avnet.py`**: **Partially Working.** Contains the core 1D-CNN + GRU architecture. However, it currently predicts both velocity and attitude (as a 4D quaternion) simultaneously using a hard-coded 200-sample window.
*   **`AI_Model/dataset_loader.py`**: **Partially Working.** Implements basic CSV parsing for `IO-VNBD` (extracting Accel, Gyro, and Vehicle Velocity) and slides a window over the data.
*   **`AI_Model/train.py`**: **Placeholder.** A basic PyTorch training loop. It lacks validation, uses random window splits (which leaks sequence data), and lacks data normalization.
*   **`docs/IMPLEMENTATION_VALIDATION.md`**: **Working.** Contains the dataset validation report.
*   **`Edge_Engine/`, `Filter_Engine/`, `Mobile_App/`**: **Placeholders.** Currently empty directories.

## CONFLICTS (Existing Implementation vs. Master Specification)

*   **Repository Structure:** Existing code is in `AI_Model/`. The master document mandates a production structure (`ml/`, `libnavik/`, `scripts/`, `configs/`, etc.).
*   **Model Architecture:** `avnet.py` combines speed and attitude prediction. The master document decouples these into **SpeedNet (M3)** and **AttitudeNet (M4)**.
*   **Data Slicing:** `train.py` uses random sliding windows. The master document strictly prohibits randomly splitting overlapping windows to avoid sequence leakage.
*   **Window Size:** `avnet.py` hard-codes 200 samples. Given `IO-VNBD`'s 10 Hz sampling rate, 200 samples = 20 seconds, which contradicts the goal of real-time instantaneous velocity prediction.
*   **Attitude Supervision:** `avnet.py` attempts to supervise attitude. The master document requires verifying ground truth first (which `IO-VNBD` lacks for pitch/roll).

## MISSING (Required Modules Not Implemented)

*   **Data Pipeline:** Automated V+S session inventory, timestamp verification, global normalization statistics generation, and manifest creation.
*   **Classical Baseline:** Python reference implementation of IMU strapdown integration and Error-State Kalman Filter (ESKF) with ZUPT, ZIHR, and NHC.
*   **C++ Navigation Engine (`libnavik`):** The core production implementation of SO(3), SE_2(3), ESKF, and InEKF.
*   **Map Matching Pipeline:** Offline OSM processing and HMM (Viterbi) implementation.
*   **Auxiliary ML Models:** AlignmentNet, Motion State classifier (M5), DenoiseNet (M2), and the Noise Adapter (M6).
*   **Evaluation Harness:** Framework to test 50m to 2000m GNSS outages and compute horizontal translation error.
*   **Mobile App & UI:** Live map, sensor dashboard, and GNSS/DR/FUSED mode manager.

## PHASE 1 — RESTRUCTURE
*   **What was moved:** `AI_Model/avnet.py` -> `ml/models/speednet/avnet_legacy.py`, `AI_Model/dataset_loader.py` -> `ml/data/dataset_loader.py`, `AI_Model/train.py` -> `ml/training/train_legacy.py`.
*   **What was preserved:** All logic inside the moved files.
*   **New directory structure:** Created `data/`, `ml/`, `libnavik/`, `map/`, `configs/`, `scripts/`, `tests/` per master specification. Created placeholder READMEs in model subdirectories. Created minimal CMake scaffolding in `libnavik/`.
*   **Compatibility decisions:** `AI_Model` directory was retained with tiny wrapper scripts (`avnet.py`, `dataset_loader.py`, `train.py`) that forward imports to the new `ml/` locations so existing calls do not instantly break. `train_legacy.py` import paths were updated.
*   **Files intentionally left untouched:** `docs/IMPLEMENTATION_VALIDATION.md` remains unchanged as the historical record of Phase 0.
*   **Files still missing:** The concrete implementations of `libnavik` interfaces, `SpeedNet`, `AttitudeNet`, data normalization, and ESKF are deliberately pending subsequent phases.

## PHASE 2 — FREEZE INTERFACES
*   **Interfaces created:** Defined `navik::ISensorSource`, `navik::IModelRunner`, `navik::INavikFilter`, `navik::IMapMatcher`, and `navik::NavikEngine` in `libnavik/include/navik/interfaces.hpp`.
*   **Types created:** Defined `ImuSample`, `MagSample`, `GnssSample`, `NavState`, `MotionState`, and `SpeedMeasurement` in `libnavik/include/navik/types.hpp`.
*   **Contracts documented:** Created `docs/contracts/app_engine_contract.md` defining the bidirectional JSON schema between Mobile App and Engine.
*   **Configurations:** Initialized generic configuration in `configs/engine_config.json`.
*   **ADR:** Authored `docs/adr/0001-interface-contracts.md` establishing coordinate frames (s-frame, v-frame, n-frame ENU), timestamps (Unix ms), and units.
*   **Tests Run:** Compiled and executed `libnavik/tests/test_interfaces.cpp` successfully using `g++` (C++14 fallback compatible using output parameters instead of `std::optional`). 

## RISKS (Technical Assumptions Requiring Validation)

*   **AttitudeNet Feasibility:** `IO-VNBD` only provides `Heading` and `Yaw Rate` from the vehicle CAN bus. Supervising full 3D attitude (pitch/roll) is impossible without fabricating data. AttitudeNet must be strictly adapted for yaw-only, or dropped in favor of classical gravity-based pitch/roll estimation.
*   **Sensor Alignment:** The dataset assumes the smartphone is fixed, but vibrations and exact mount angles may vary. The analytic alignment (M1) must be robust enough to handle this before neural networks can generalize.
*   **Time Synchronization:** While `Synchronised V and S datasets` exist, verifying the absolute accuracy of this synchronization is critical before training SpeedNet. Minor millisecond offsets could misalign acceleration spikes with CAN-bus velocity changes.

## NEXT (Exact Implementation Order)

1.  **PHASE 1:** Restructure the repository to match the master specification (`ml/`, `libnavik/`, `data/`, etc.). Move `avnet.py` to `ml/models/speednet.py`.
2.  **PHASE 2:** Freeze canonical interfaces (create `libnavik/include/navik/types.hpp`).
3.  **PHASE 3:** Build the robust `IO-VNBD` Data Pipeline (inventory, sync verification, scaling, train/val splitting without leakage).
4.  **PHASE 4:** Implement the Classical Baseline (Strapdown + ESKF with Oracle Speed). This is a strict blocking gate.
5.  **PHASE 5:** Refactor AVNet into `SpeedNet` and train it on the validated data pipeline.
