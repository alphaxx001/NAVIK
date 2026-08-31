package com.sih.idrsystem

import kotlin.math.cos
import kotlin.math.sin

enum class NavMode {
    GNSS, DEAD_RECKONING, MAP_CONTEXT, DR_FALLBACK
}

class NavigationEngine {
    var mode = NavMode.GNSS
    var lat = 0.0
    var lon = 0.0
    var heading = 0.0 // rad
    
    // Earth radius
    private val R = 6378137.0
    
    // True if user toggles GNSS OFF for demo
    var simulateGnssOutage = false
    var mapConfidenceHigh = false

    fun init(startLat: Double, startLon: Double, startHeading: Double) {
        lat = startLat
        lon = startLon
        heading = startHeading
    }

    fun step(
        dt: Double,
        speedMps: Float,
        probStationary: Float,
        yawRateRadS: Float,
        gnssValid: Boolean,
        gnssLat: Double,
        gnssLon: Double
    ) {
        val isGnssAvailable = gnssValid && !simulateGnssOutage
        
        if (isGnssAvailable) {
            mode = NavMode.GNSS
            lat = gnssLat
            lon = gnssLon
            // simple heading update from model for smooth display, or could use GNSS heading
        } else {
            // Check Map Context
            if (mapConfidenceHigh) {
                mode = NavMode.MAP_CONTEXT
            } else {
                mode = NavMode.DEAD_RECKONING
            }
            
            // Apply ZUPT
            val effectiveSpeed = if (probStationary > 0.569f) 0.0 else speedMps.toDouble()
            val effectiveYaw = if (probStationary > 0.569f) 0.0 else yawRateRadS.toDouble()
            
            // Simple Kinematic Integration (Placeholder for 15-state ESKF)
            heading += effectiveYaw * dt
            
            // Advance Position
            val dDistance = effectiveSpeed * dt
            val dLat = dDistance * cos(heading) / R
            val dLon = dDistance * sin(heading) / (R * cos(Math.toRadians(lat)))
            
            lat += Math.toDegrees(dLat)
            lon += Math.toDegrees(dLon)
        }
    }
}
