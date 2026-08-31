import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Edge_Engine.runtime.sensor_buffer import SensorBuffer
from Edge_Engine.runtime.model_runner import ModelRunner
from Edge_Engine.runtime.state_machine import NavigationStateMachine

def test_sensor_buffer():
    sb = SensorBuffer(window_size=200)
    for i in range(199):
        sb.add_sample(float(i), np.array([0,0,9.8]), np.array([0,0,0]))
        
    assert not sb.is_ready()
    
    sb.add_sample(200.0, np.array([0,0,9.8]), np.array([0,0,0]))
    assert sb.is_ready()
    
    w = sb.get_window()
    assert w.shape == (6, 200)

def test_model_runner():
    mr = ModelRunner()
    w = np.zeros((6, 200))
    s, m, y = mr.run_inference(w)
    assert isinstance(s, float)
    assert isinstance(m, float)
    assert isinstance(y, float)

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
