# ADR 0001: Interface Contracts and Data Conventions

## Status
Accepted

## Context
To enable parallel development across the Data/ML team, the Navigation/Filter team, and the Mobile/Edge team, we must establish strict interface contracts. The navigation algorithms (ESKF, InEKF) and neural networks (SpeedNet) must communicate through strongly typed C++ interfaces without being tightly coupled to the specifics of Android sensors or PyTorch datasets.

## Decision

1. **Freezing Interfaces Before Implementation**: We defined `types.hpp` and `interfaces.hpp` in `libnavik` to outline exactly how modules pass data, before any complex algorithms are written.
2. **Coordinate Frame Conventions**:
   - `s-frame` (Sensor Frame): Right-Handed (x=Right, y=Forward, z=Up).
   - `v-frame` (Vehicle Frame): Right-Handed (x=Right, y=Forward, z=Up).
   - `n-frame` (Navigation Frame): East-North-Up (ENU).
3. **Timestamp Convention**:
   - `timestamp_ms`: `uint64_t` representing milliseconds since Unix Epoch.
4. **Unit Conventions**:
   - Acceleration: meters per second squared (m/s²).
   - Angular Velocity: radians per second (rad/s).
   - Speed/Velocity: meters per second (m/s).
   - Position (GNSS): Degrees for Lat/Lon, meters for altitude.
5. **Ownership**:
   - `libnavik/include/navik/types.hpp` is the absolute source of truth for the C++ engine.
   - `docs/contracts/app_engine_contract.md` is the absolute source of truth for Edge-to-Mobile communication.

## Consequences
- The ML team must ensure their ONNX/TFLite exported models accept window sizes and channel counts that exactly match `IModelRunner` inputs.
- The Android team can immediately begin building UI based on the `app_engine_contract.md` JSON schema using mock data.
- The Filter team can build the ESKF entirely using `ImuSample` and `GnssSample` without writing Android or CSV data ingestion logic.
