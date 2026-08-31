package com.sih.idrsystem

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

class ModelRunner(
    speedNetPath: String,
    motionNetPath: String,
    headingNetPath: String
) {
    private val env = OrtEnvironment.getEnvironment()
    private val speedSession: OrtSession
    private val motionSession: OrtSession
    private val headingSession: OrtSession
    
    // Normalization constants (Hardcoded from normalization.json for prototype, normally loaded from JSON)
    private val featuresMean = floatArrayOf(-0.023626f, -0.00899f, 9.8518f, 0.00021f, -0.007565f, 0.001232f)
    private val featuresStd = floatArrayOf(1.3567f, 1.3206f, 0.6512f, 0.1146f, 0.1937f, 0.1080f)
    
    private val speedMu = 10.4019f
    private val speedStd = 7.2721f
    private val yawScaler = 0.2f

    init {
        // Load ONNX sessions
        val options = OrtSession.SessionOptions()
        speedSession = env.createSession(speedNetPath, options)
        motionSession = env.createSession(motionNetPath, options)
        headingSession = env.createSession(headingNetPath, options)
    }

    fun setNormalization(mean: FloatArray, std: FloatArray) {
        System.arraycopy(mean, 0, featuresMean, 0, 6)
        System.arraycopy(std, 0, featuresStd, 0, 6)
    }

    fun runInference(windowFlat: FloatArray): Triple<Float, Float, Float> {
        // Normalize the flat window
        val normalized = FloatArray(windowFlat.size)
        for (c in 0 until 6) {
            val mu = featuresMean[c]
            val std = featuresStd[c]
            for (i in 0 until 20) {
                val idx = c * 20 + i
                normalized[idx] = (windowFlat[idx] - mu) / std
            }
        }

        val shape = longArrayOf(1, 6, 20)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(normalized), shape)
        val inputs = Collections.singletonMap("input", tensor)

        // Run inferences
        val speedResult = speedSession.run(inputs)
        val motionResult = motionSession.run(inputs)
        val headingResult = headingSession.run(inputs)

        // Extract values
        val sRaw = (speedResult[0].value as Array<FloatArray>)[0][0]
        val mRaw = (motionResult[0].value as Array<FloatArray>)[0][0]
        val hRaw = (headingResult[0].value as Array<FloatArray>)[0][0]

        // Denormalize
        var speedMps = sRaw * speedStd + speedMu
        var prob = mRaw
        var yawRad = hRaw * yawScaler

        if (speedMps.isNaN() || speedMps.isInfinite()) speedMps = 0f
        if (yawRad.isNaN() || yawRad.isInfinite()) yawRad = 0f
        prob = prob.coerceIn(0f, 1f)

        speedResult.close()
        motionResult.close()
        headingResult.close()
        tensor.close()

        return Triple(speedMps, prob, yawRad)
    }
}
