from model_core.data_loader_ashare import AShareDataLoader
import torch

def test():
    print("Testing AShareDataLoader...")
    # Use a limit to speed up test
    loader = AShareDataLoader(data_dir='data/csv')
    loader.load_data(limit_stocks=50) # Load 50 stocks for testing
    
    print("Keys in raw_data:", loader.raw_data_cache.keys())
    print("Feature Tensor Shape:", loader.feat_tensor.shape)
    print("Target Tensor Shape:", loader.target_ret.shape)
    
    # Check for NaNs
    if torch.isnan(loader.feat_tensor).any():
        print("WARNING: NaNs in features!")
    else:
        print("Features clean.")

if __name__ == "__main__":
    test()
