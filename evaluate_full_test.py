import argparse
from run import Options

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on full test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model zip file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Evaluating model on ENTIRE test set (July 2018 - July 2019)")
    print("="*60 + "\n")
    
    model = Options().get_model(args.model_path)
    
    # Run inference on entire test set - this will print both metrics
    model.Inference()
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
