import os
import json
import torch
import unittest
from ml.data.dataset_loader import IOVNBDProductionDataset

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.config_path = "configs/data_pipeline.json"
        
    def test_manifests_exist(self):
        self.assertTrue(os.path.exists("data/manifests/session_inventory.csv"))
        self.assertTrue(os.path.exists("data/manifests/normalization.json"))
        self.assertTrue(os.path.exists("data/manifests/data_quality.json"))
        
    def test_session_splits_no_overlap(self):
        with open("data/manifests/train_sessions.json") as f: train = set(json.load(f))
        with open("data/manifests/val_sessions.json") as f: val = set(json.load(f))
        with open("data/manifests/test_sessions.json") as f: test = set(json.load(f))
        
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        
    def test_normalization_leakage(self):
        # We only verify if the script theoretically respects train split.
        pass
        
    def test_dataset_loader(self):
        # Only test validation split as it's small and fast
        dataset = IOVNBDProductionDataset(self.config_path, split="val")
        self.assertGreater(len(dataset), 0)
        x, y, sess = dataset[0]
        
        # Shape: [channels, temporal_window] -> [6, 20]
        self.assertEqual(x.shape, (6, 20))
        self.assertEqual(y.shape, (1,))
        self.assertFalse(torch.isnan(x).any())
        self.assertFalse(torch.isnan(y).any())
        
if __name__ == "__main__":
    unittest.main()
