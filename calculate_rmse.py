import argparse
import numpy as np
from run import Options
from utils.metric import RMSE, MAPE

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate RMSE and MAPE for a prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model zip file")
    parser.add_argument("--test_time", type=str, required=True, help="Test time in the format YYYY-MM-DD HH:MM:SS")
    return parser.parse_args()

def main():
    args = parse_args()

    model = Options().get_model(args.model_path)
    
    # get prediction and ground truth
    predicted, ground_truth = model.test_single(args.test_time)
    
    # calculate metrics
    rmse = RMSE(predicted, ground_truth)
    mape = MAPE(predicted, ground_truth)
    
    print("\n" + "="*50)
    print(f"Metrics for prediction at {args.test_time}")
    print("="*50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
