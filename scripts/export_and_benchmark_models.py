import os
import sys
import json
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.models.speednet.speednet import SpeedNet
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.models.heading.heading_net import HeadingNet

def load_pytorch_model(model_class, config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    model = model_class(
        in_channels=cfg['model']['in_channels'],
        cnn_channels=cfg['model']['cnn_channels'],
        kernel_size=cfg['model']['kernel_size'],
        gru_hidden=cfg['model']['gru_hidden'],
        gru_layers=cfg['model']['gru_layers']
    )
    checkpoint_path = cfg['paths']['best_model']
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

def benchmark_and_export():
    os.makedirs("models/onnx", exist_ok=True)
    
    models = {
        "SpeedNet": (SpeedNet, "configs/speednet_training.json"),
        "MotionStateNet": (MotionStateNet, "configs/motion_state_training.json"),
        "HeadingNet": (HeadingNet, "configs/heading_training.json")
    }
    
    dummy_input = torch.randn(1, 6, 20, dtype=torch.float32)
    results = []
    
    for name, (model_cls, cfg_path) in models.items():
        print(f"--- Processing {name} ---")
        model = load_pytorch_model(model_cls, cfg_path)
        
        # PyTorch Inference
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
            
        # ONNX Export
        onnx_path = f"models/onnx/{name}.onnx"
        export_success = False
        try:
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path, 
                export_params=True, 
                opset_version=11, 
                do_constant_folding=True, 
                input_names=['input'], 
                output_names=['output'], 
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_out = ort_session.run(None, ort_inputs)[0]
            
            max_diff = np.max(np.abs(pt_out - ort_out))
            print(f"Max difference (PyTorch vs ONNX): {max_diff}")
            export_success = True
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            
        # Latency Benchmark (PyTorch)
        # Warmup
        for _ in range(50):
            _ = model(dummy_input)
            
        latencies = []
        for _ in range(200):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            
        # Latency Benchmark (ONNX)
        onnx_latencies = []
        if export_success:
            for _ in range(50):
                _ = ort_session.run(None, ort_inputs)
            for _ in range(200):
                t0 = time.perf_counter()
                _ = ort_session.run(None, ort_inputs)
                t1 = time.perf_counter()
                onnx_latencies.append((t1 - t0) * 1000)
                
        results.append({
            "name": name,
            "export_success": export_success,
            "pt_mean_ms": np.mean(latencies),
            "pt_p95_ms": np.percentile(latencies, 95),
            "onnx_mean_ms": np.mean(onnx_latencies) if export_success else None,
            "onnx_p95_ms": np.percentile(onnx_latencies, 95) if export_success else None,
            "max_diff": max_diff if export_success else None
        })
        
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    benchmark_and_export()
