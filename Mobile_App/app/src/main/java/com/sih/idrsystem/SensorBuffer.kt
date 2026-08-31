package com.sih.idrsystem

class SensorBuffer(private val windowSize: Int = 20) {
    private val buffer = mutableListOf<FloatArray>()
    
    // Each sample is expected to be [ax, ay, az, gx, gy, gz]
    fun addSample(sample: FloatArray) {
        if (sample.size != 6) throw IllegalArgumentException("Expected 6 channels")
        buffer.add(sample)
        if (buffer.size > windowSize) {
            buffer.removeAt(0)
        }
    }
    
    fun isReady(): Boolean = buffer.size == windowSize
    
    // Returns [6, 20] array mapped to 1D float array for ONNX
    fun getWindowFlat(): FloatArray {
        val flat = FloatArray(6 * windowSize)
        for (channel in 0 until 6) {
            for (i in 0 until windowSize) {
                flat[channel * windowSize + i] = buffer[i][channel]
            }
        }
        return flat
    }
    
    fun clear() {
        buffer.clear()
    }
}
