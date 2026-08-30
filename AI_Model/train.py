# Legacy compatibility wrapper
from ml.training.train_legacy import train_model

if __name__ == '__main__':
    # We can't trivially execute this wrapper seamlessly if the paths are messed up
    # but the function is exported.
    pass
