package com.sih.idrsystem

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ReplayTest {

    @Test
    fun testPipelineStructure() {
        val buffer = SensorBuffer()
        val engine = NavigationEngine()
        
        // 1. Initial State
        engine.init(12.0, 77.0, 0.0)
        assertEquals(NavMode.GNSS, engine.mode)
        
        // 2. Feed Samples
        for (i in 0 until 19) {
            buffer.addSample(floatArrayOf(0f, 0f, 9.8f, 0f, 0f, 0f))
            assertFalse(buffer.isReady())
        }
        
        buffer.addSample(floatArrayOf(0f, 0f, 9.8f, 0f, 0f, 0f))
        assertTrue(buffer.isReady())
        
        val flat = buffer.getWindowFlat()
        assertEquals(6 * 20, flat.size)
        
        // 3. Simulated Model Runner
        // (Cannot easily run ONNX in JVM local unit tests without downloading specific JNI binaries for Windows/Linux in Gradle,
        // so we bypass the strict ONNX execution for this logic test and test the Navigation Engine state machine).
        
        val speedMps = 15.0f
        val prob = 0.1f
        val yaw = 0.05f
        
        // 4. GNSS OFF causes transition
        engine.simulateGnssOutage = true
        engine.step(0.1, speedMps, prob, yaw, true, 12.0, 77.0)
        assertEquals(NavMode.DEAD_RECKONING, engine.mode)
        
        // 5. State advances
        assertNotEquals(12.0, engine.lat, 0.0000001)
        
        // 6. GNSS Restoration
        engine.simulateGnssOutage = false
        engine.step(0.1, speedMps, prob, yaw, true, 12.01, 77.01)
        assertEquals(NavMode.GNSS, engine.mode)
        assertEquals(12.01, engine.lat, 0.0000001)
    }
}
