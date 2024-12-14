import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from utils.data_loader import load_data
from feature_selection.deep_lasso import DeepLasso

def train_model(dataset, model, regulation=None, topk=None):
    print(f"Training {model} on {dataset} with regulation={regulation}, topk={topk}.")
    data = load_data(dataset)
    if regulation == "deep_lasso":
        selector = DeepLasso()
        features = selector.select(data, topk)
        # Training logic goes here
    else:
        # Training without feature selection
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["mlp", "ft_transformer"], required=True)
    parser.add_argument("--regulation", type=str, choices=["deep_lasso"], default=None)
    parser.add_argument("--topk", type=float, help="Top K features to select", default=None)
    args = parser.parse_args()

    train_model(args.dataset, args.model, args.regulation, args.topk)
