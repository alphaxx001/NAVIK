import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from scripts.test_kdtree_equivalence import KDTreeHMMMapMatcher

def enu_to_latlon(e, n, lat0, lon0):
    R = 6378137.0
    lat0_rad = np.radians(lat0)
    lat = lat0 + np.degrees(n / R)
    lon = lon0 + np.degrees(e / (R * np.cos(lat0_rad)))
    return lat, lon


from Edge_Engine.runtime.sensor_buffer import SensorBuffer
from Edge_Engine.runtime.model_runner import ModelRunner
from Edge_Engine.runtime.state_machine import NavigationStateMachine
from Edge_Engine.runtime.navigation_output import NavigationOutput

class EdgeEngine:
    def __init__(self, map_graph_file):
        self.sensor_buffer = SensorBuffer(window_size=20)
        self.model_runner = ModelRunner(device_str="cpu")
        self.state_machine = NavigationStateMachine()
        self.eskf = ESKF()
        self.matcher = KDTreeHMMMapMatcher(map_graph_file, search_radius=100.0)
        
        self.is_initialized = False
        self.lat0 = 0.0
        self.lon0 = 0.0
        
        self.last_gnss_time = 0
        self.last_imu_time = 0
        
        # Hysteresis for motion state
        self.motion_history = []
        self.trajectory_buffer = []
        self.inference_freq = 1 # Process NNs every IMU sample (10Hz)
        self.sample_count = 0
        
    def initialize_gnss(self, lat, lon, heading_deg, timestamp):
        self.lat0 = lat
        self.lon0 = lon
        yaw_rad = (90.0 - heading_deg) * np.pi / 180.0
        self.eskf.q = R.from_euler('z', yaw_rad)
        self.eskf.p = np.zeros(3)
        self.eskf.v = np.zeros(3)
        self.is_initialized = True
        self.last_gnss_time = timestamp
        self.last_imu_time = timestamp
        
    def step_imu(self, timestamp, accel, gyro):
        self.sample_count += 1
        self.sensor_buffer.add_sample(timestamp, accel, gyro)
        
        dt = timestamp - self.last_imu_time
        if dt <= 0: dt = 0.01
        self.last_imu_time = timestamp
        
        if not self.is_initialized:
            return None
            
        p_gyro = gyro.copy()
        run_inference = (self.sample_count % self.inference_freq == 0) and self.sensor_buffer.is_ready()
        
        speed_mps = 0.0
        motion_prob = 0.0
        yaw_rate = 0.0
        
        if run_inference:
            window = self.sensor_buffer.get_window()
            speed_mps, motion_prob, yaw_rate = self.model_runner.run_inference(window)
            p_gyro[2] = yaw_rate # Override Z-gyro with HeadingNet
            
            # Hysteresis update
            is_stat = 1 if motion_prob > 0.569 else 0
            self.motion_history.append(is_stat)
            if len(self.motion_history) > 3:
                self.motion_history.pop(0)
                
        # 1. Authoritative Prediction
        self.eskf.predict(accel, p_gyro, dt)
        
        if run_inference:
            # 2. ESKF Updates (Authoritative physics only)
            if sum(self.motion_history) == 3:
                self.eskf.update_zupt()
            else:
                self.eskf.update_oracle_speed(speed_mps, nhc=True)
                
            # Track for HMM
            self.trajectory_buffer.append({'x': self.eskf.p[0], 'y': self.eskf.p[1], 'h': self.eskf.q.as_euler('xyz')[2]})
            if len(self.trajectory_buffer) > 50:
                self.trajectory_buffer.pop(0) # Keep sliding window for Viterbi
                
        return None # Outputs only computed at request or inference step
        
    def generate_output(self, gnss_available):
        dr_p = self.eskf.p.copy()
        map_p = None
        has_candidates = False
        fallback_reason = None
        candidate_count = 0
        
        if len(self.trajectory_buffer) > 0:
            cands = self.matcher.get_candidates_kdtree(dr_p[0], dr_p[1], self.eskf.q.as_euler('xyz')[2])
            candidate_count = len(cands)
            if candidate_count > 0:
                has_candidates = True
                
                # In a real online streaming system we just use the latest Viterbi state
                # We can approximate the Viterbi output by running match on the buffer
                matched_pts, states = self.matcher.viterbi_match(self.trajectory_buffer)
                
                if states[-1] != 'DR_FALLBACK':
                    map_p = matched_pts[-1]
                else:
                    has_candidates = False
                    fallback_reason = "Viterbi Transition Penalty Rejected Topology"
            else:
                fallback_reason = "No KD-Tree candidates within 100m"
                
        self.state_machine.update(gnss_available, has_candidates)
        mode = self.state_machine.get_state()
        
        final_x, final_y = dr_p[0], dr_p[1]
        
        if mode == "MAP_CONTEXT" and map_p is not None:
            final_x, final_y = map_p[0], map_p[1]
            
        out_lat, out_lon = enu_to_latlon(final_x, final_y, self.lat0, self.lon0)
        
        return NavigationOutput(
            timestamp=self.last_imu_time,
            mode=mode,
            latitude=out_lat,
            longitude=out_lon,
            enu_position=(final_x, final_y),
            speed_mps=np.linalg.norm(self.eskf.v),
            heading=self.eskf.q.as_euler('xyz')[2],
            gnss_available=gnss_available,
            stationary_probability=0.0, # Approximate here unless stored
            map_confidence=0.9 if mode == "MAP_CONTEXT" else 0.0,
            map_candidate_count=candidate_count,
            fallback_reason=fallback_reason,
            authoritative_dr_position=(dr_p[0], dr_p[1]),
            map_context_position=(map_p[0], map_p[1]) if map_p is not None else None
        )
