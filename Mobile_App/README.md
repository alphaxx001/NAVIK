# IDR-System Mobile App

This module contains the Android/Kotlin mobile prototype for the SIH navigation system. 
It wraps the validated Deep Learning (ONNX) + ESKF + Map Matching navigation core.

## Stack
- **Platform:** Android (Native)
- **Language:** Kotlin
- **ML Inference:** ONNX Runtime for Android (`onnxruntime-android`)
- **Map Visualization:** Mapbox GL / OSMDroid (Planned)
- **Architecture:** MVVM + Clean Architecture

## Directory Structure
- `app/src/main/java/com/sih/idrsystem/` : Core Kotlin logic (Sensors, ONNX wrappers, State Machine)
- `app/src/main/res/` : UI Layouts
- `app/src/main/assets/` : ONNX models and offline Map datasets
