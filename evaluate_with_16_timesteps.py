import argparse
from run import Options
from utils.metric import metric_rolling

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate with rm=16 (true 8 hours)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model zip file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Evaluating with CORRECTED metric (rm=16 for true 8 hours)")
    print("="*60 + "\n")
    
    model = Options().get_model(args.model_path)
    
    # Get all predictions
    import pandas as pd
    test_set = pd.read_csv("./data_provider/datasets/test_timestamps_24avg.tsv", sep="\t")
    test_points = test_set["Hold Out Start"]
    pre = []
    gt = []
    
    print("Running predictions on all test timestamps...")
    for i, testP in enumerate(test_points):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_points)}")
        predicted, ground_truth = model.test_single(testP)
        pre.extend(predicted)
        gt.extend(ground_truth)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    
    # 3-day prediction (full 72 hours)
    metric_rolling("Every 3 days (72 hours)", pre, gt, rm=72, inter=72)
    
    # TRUE 8-hour prediction (16 timesteps)
    print("\n")
    metric_rolling("Every 8 hours (rm=16, TRUE 8 hours)", pre, gt, rm=16, inter=72)
    
    # What the code currently does (4 hours)
    print("\n")
    metric_rolling("Every 4 hours (rm=8, what code does)", pre, gt, rm=8, inter=72)
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
