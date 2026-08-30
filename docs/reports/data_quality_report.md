# IO-VNBD Data Quality Report

## Overview
This document summarizes the automated quality checks and ingestion of the IO-VNBD dataset. The pipeline strictly analyzes raw files in `data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/` without modifying their source history or contents.

## Key Findings

### Dataset Inventory & Validations
- **Discovered Files:** 72 Smartphone (`S-*`) files and 72 Vehicle (`V-*`) files.
- **Valid Synchronized Sessions:** 63 pairs match perfectly in row counts and display stable timestamp alignment.
- **Invalid / Excluded Sessions:** 9 sessions were excluded due to differing row counts between the V and S files, indicating potential synchronization breaks (these were ignored to maintain sequence integrity).

### Sampling Rate Analysis
- **Measured Hz:** The nominal frequency, measured empirically through `np.diff()` on the timestamp columns (`TIME SINCE START (ms)`), is **9.97 Hz**. 
- **Assumption Overridden:** Previous hypotheses assumed 100 Hz. The measured ~10 Hz requires adapting model temporal windows. A standard 2.0-second window correctly contains **20 samples**.

### Character Encoding Anomalies
- **Unicode Errors Resolved:** Initial loading crashed due to non-UTF8 bytes (specifically the `²` symbol in `m/s²`). This was resolved by forcing `latin1` decoding across all IO-VNBD files, ensuring raw sensor fields load without stripping columns or corrupting numeric data.

### Fields and Unit Normalization
- **Acceleration:** Extracted `ACCELEROMETER X/Y/Z (m/s²)`. Internal unit strictly maintained as `m/s²`.
- **Gyroscope:** Extracted `GYROSCOPE X/Y/Z (rad/s)`. Internal unit strictly maintained as `rad/s`.
- **Velocity Ground Truth:** Extracted from the CAN-Bus `Velocity (km/hr)` and internally converted to `m/s` directly inside the ingestion logic (`velocity / 3.6`).
- **Attitude Supervision:** Re-confirmed: Pitch/Roll is *not* present in the dataset, thus `AttitudeNet` cannot be trained on this specific dataset without adapting to yaw-only.

### NaN / Gap Handling
- Minimal NaN gaps were detected. These were handled via linear interpolation combined with boundary fill (`bfill` + `ffill`), which respects causality better than simple mean imputation.

### Data Splits (Session-Level)
- **Train:** 46 distinct sessions
- **Validation:** 5 distinct sessions
- **Test:** 12 distinct sessions
- *Strict Separation:* Overlapping window generation was disabled for dataset partitioning. Entire continuous sessions were grouped into distinct sets.

### Visual Sanity Checks
- Plots generated in `ml/data/outputs/sanity_check_*.png` display cohesive alignment between sharp velocity changes (CAN bus) and acceleration spikes (Smartphone), confirming that row-alignment equates to temporal alignment.
