"""
Complete pipeline runner - executes all steps in sequence.
Use this script to run the entire workflow from data preparation to evaluation.
"""
import sys
from pathlib import Path

def main():
    print("="*60)
    print("Oilseed Plant Classification - Complete Pipeline")
    print("="*60)
    
    # Step 1: Prepare data
    print("\n[1/4] Preparing dataset...")
    try:
        from prepare_data import split_dataset
        split_dataset()
    except Exception as e:
        print(f"Error in data preparation: {e}")
        sys.exit(1)
    
    # Step 2: Train model
    print("\n[2/4] Training model...")
    try:
        from train import train
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error in training: {e}")
        sys.exit(1)
    
    # Step 3: Evaluate model
    print("\n[3/4] Evaluating model...")
    try:
        from evaluate import evaluate_model
        evaluate_model()
    except Exception as e:
        print(f"Error in evaluation: {e}")
        sys.exit(1)
    
    # Step 4: Print summary
    print("\n[4/4] Pipeline complete!")
    print("\n" + "="*60)
    print("Next steps:")
    print("  - Check results/ folder for training curves and confusion matrix")
    print("  - Use inference.py to predict on new images:")
    print("    python inference.py path/to/image.jpg")
    print("="*60)

if __name__ == "__main__":
    main()



