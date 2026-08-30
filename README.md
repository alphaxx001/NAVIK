# NAVIK — Navigation Assurance in GNSS-denied Indian Kinematics

## Problem
SIH 26168: AI-ML based Intelligent Dead Reckoning (IDR) system for seamless navigation. The goal is to track vehicle motion accurately using smartphone IMU sensors during GNSS outages (e.g., tunnels, urban canyons).

## High-Level Architecture
NAVIK relies on a hybrid Cloud-to-Edge architecture incorporating:
- **Data-Driven Measurements:** `SpeedNet` (1D-CNN + GRU) to estimate forward velocity from raw IMU, bypassing the exponential drift of double integration.
- **Model-Driven Filtering:** An Error-State Kalman Filter (ESKF), eventually upgrading to an Invariant EKF (InEKF), to fuse GNSS and AI-measurements with physical constraints (NHC, ZUPT, ZIHR).
- **Map Matching:** Offline OSM processing and HMM (Viterbi) to snap the estimated trajectory onto road geometries.
- **Mobile/Edge Engine:** A C++ backend (`libnavik`) and an Android/Flutter UI to provide offline real-time navigation.

## Current Implementation Status
We have completed **Phase 1 (Repository Restructure)**.
- Legacy prototype code has been preserved in `ml/`.
- The production directory structure has been created.
- The project is blocked from full model training until the data pipeline and classical baseline (ESKF) are strictly validated.

## Build Order
1.  **Data Pipeline:** IO-VNBD dataset synchronization and normalization.
2.  **Classical Baseline:** ESKF with Oracle vehicle speed.
3.  **SpeedNet:** Train AI model to replace oracle speed.
4.  **AttitudeNet & Motion State:** Auxiliary AI modules.
5.  **Filter Integration:** Full ESKF + AI integration + Ablation tests.
6.  **Map Matching:** HMM offline map matching.
7.  **libnavik & Mobile App:** Port to C++ and Android.

## Repository Structure
*   `docs/`: Architecture decisions (ADRs) and reports.
*   `data/`: Raw IO-VNBD datasets and processed manifests.
*   `ml/`: PyTorch models, data loaders, training scripts.
    *   `models/`: `speednet`, `attitudenet`, `denoiser`, etc.
*   `libnavik/`: The C++17 production navigation engine.
*   `map/`: OpenStreetMap preprocessing and matching.
*   `Mobile_App/` & `Edge_Engine/`: Platform-specific targets.
*   `configs/`: Hyperparameters and system configs.
*   `scripts/`: Utility scripts.
*   `tests/`: Integration tests.
