# Model Inference Preprocessing & Output Contract

This contract formally documents the frozen architectural expectations, array dimensions, physical units, and bounds for the IDR-System machine learning models.

## 1. Global Preprocessing (SensorBuffer)
All neural networks in this architecture consume identical input tensors generated from a temporal sliding window.

- **Window Duration:** $2.0$ seconds
- **Sampling Frequency:** $10$ Hz
- **Sequence Length (N):** $20$ samples
- **Channel Ordering (6):** `[Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]`
- **Raw Sensor Units:** Accelerometer ($m/s^2$), Gyroscope ($rad/s$)
- **Required Shape:** `[Batch, 6, 20]`
- **Normalization Strategy:** Z-Score Standard Normalization $((X - \mu) / \sigma)$
- **Source of Truth for $\mu$ and $\sigma$:** `data/manifests/normalization.json`

---

## 2. SpeedNet
**Architecture:** 1D-CNN feature extractor + GRU temporal aggregator + Linear Regressor.
- **Input Tensor:** `[B, 6, 20]` normalized float32
- **Output Tensor:** `[B, 1]` float32
- **Raw Model Output:** Dimensionless Z-Score mapped to target velocity distribution.
- **Denormalization:** `Speed_mps = (Output * 7.2721) + 10.4019`
- **Final Physical Unit:** $m/s$
- **Expected Range:** $[0, \sim 35]$

---

## 3. MotionStateNet
**Architecture:** 1D-CNN feature extractor + GRU temporal aggregator + Linear Classifier (Sigmoid mapped).
- **Input Tensor:** `[B, 6, 20]` normalized float32
- **Output Tensor:** `[B, 1]` float32
- **Raw Model Output:** Direct Probability. No denormalization required.
- **Final Physical Meaning:** Probability that the vehicle is stationary (ZUPT valid).
- **Expected Range:** $[0.0, 1.0]$
- **Post-Processing (Edge Engine):** Trigger ZUPT if threshold $> 0.569$ for $3$ consecutive steps.

---

## 4. HeadingNet
**Architecture:** 1D-CNN feature extractor + GRU temporal aggregator + Linear Regressor.
- **Input Tensor:** `[B, 6, 20]` normalized float32
- **Output Tensor:** `[B, 1]` float32
- **Raw Model Output:** Dimensionless Z-Score mapped to target yaw-rate distribution.
- **Denormalization:** `YawRate_rad_s = Output * 0.2` (Target mean was strictly $0.0$)
- **Final Physical Unit:** $rad/s$
- **Expected Range:** $[-\sim 1.0, \sim 1.0]$
