import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Edge_Engine.runtime.sensor_buffer import SensorBuffer
from Edge_Engine.runtime.model_runner import ModelRunner
from Edge_Engine.runtime.state_machine import NavigationStateMachine

def test_sensor_buffer():
    sb = SensorBuffer(window_size=20)
    for i in range(19):
        sb.add_sample(float(i), np.array([0,0,9.8]), np.array([0,0,0]))
        
    assert not sb.is_ready()
    
    sb.add_sample(20.0, np.array([0,0,9.8]), np.array([0,0,0]))
    assert sb.is_ready()
    
    w = sb.get_window()
    assert w.shape == (6, 20)

def test_model_runner():
    mr = ModelRunner()
    
    # 1. Accept [6, 20]
    w_valid = np.zeros((6, 20))
    s, m, y = mr.run_inference(w_valid)
    assert isinstance(s, float)
    assert isinstance(m, float)
    assert isinstance(y, float)
    
    # 2 & 3. Reject incorrect shapes
    try:
        mr.run_inference(np.zeros((6, 21)))
        assert False, "Should reject incorrect sequence length"
    except ValueError:
        pass
        
    try:
        mr.run_inference(np.zeros((5, 20)))
        assert False, "Should reject incorrect channel count"
    except ValueError:
        pass
        
    # 4. No NaN/Inf
    assert not np.isnan(s) and not np.isinf(s)
    
    # 6. Probability bounds
    assert 0.0 <= m <= 1.0
    
    # 8. Deterministic Output
    w_rand = np.random.randn(6, 20)
    s1, m1, y1 = mr.run_inference(w_rand)
    s2, m2, y2 = mr.run_inference(w_rand)
    assert s1 == s2 and m1 == m2 and y1 == y2
    
    # 9. Normalization loaded
    assert mr.norm is not None
    assert 'targets' in mr.norm

def test_state_machine():
    sm = NavigationStateMachine()
    
    # Init
    assert sm.get_state() == "GNSS"
    
    # GNSS Lost, map available
    sm.update(False, True)
    assert sm.get_state() == "MAP_CONTEXT"
    
    # GNSS Lost, map unavailable
    sm.update(False, False)
    assert sm.get_state() == "DR_FALLBACK"
    
    # GNSS Returns
    sm.update(True, False)
    assert sm.get_state() == "GNSS"
    
if __name__ == "__main__":
    test_sensor_buffer()
    test_model_runner()
    test_state_machine()
    print("All Edge Engine core tests passed.")
