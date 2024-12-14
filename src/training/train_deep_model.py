import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.data_loader import load_data
from feature_selection.deep_lasso import DeepLasso


def train_model(dataset, model, regulation=None, topk=None):
    print(f"Starting training: Dataset={dataset}, Model={model}, Regulation={regulation}, Topk={topk}")

    # Step 1: Load dataset
    try:
        print("Loading dataset...")
        data = load_data(dataset)
        print(f"Dataset '{dataset}' loaded successfully with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 2: Split features and labels
    try:
        print("Splitting features and labels...")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    except Exception as e:
        print(f"Error splitting dataset into features and labels: {e}")
        return

    # Step 3: Apply feature selection
    if regulation == "deep_lasso":
        try:
            print("Applying Deep Lasso for feature selection...")
            selector = DeepLasso()
            selector.calculate_importance(X, model=None)  # Gradient-based feature selection (placeholder for now)
            X = selector.select(X, topk)
            print(f"Top-{topk} features selected. New features shape: {X.shape}")
        except Exception as e:
            print(f"Error in feature selection with Deep Lasso: {e}")
            return

    # Step 4: Train-test split
    try:
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Step 5: Train the model
    try:
        print(f"Training model '{model}'...")
        if model == "random_forest":
            regressor = RandomForestRegressor(random_state=42)
            regressor.fit(X_train, y_train)
            print("Model training completed.")

            # Step 6: Evaluate the model
            print("Evaluating the model...")
            y_pred = regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Model evaluation completed. Mean Squared Error: {mse}")
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        print(f"Error during model training or evaluation: {e}")
        return

    print("Training process completed successfully.")


if __name__ == "__main__":
    try:
        # Step 0: Parse arguments
        print("Parsing arguments...")
        parser = argparse.ArgumentParser(description="Train a model with feature selection.")
        parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
        parser.add_argument("--model", type=str, required=True, help="Model to train")
        parser.add_argument("--regulation", type=str, required=False, help="Feature selection regulation")
        parser.add_argument("--topk", type=float, required=False, help="Top-k proportion of features to select")
        args = parser.parse_args()

        print(f"Arguments received: Dataset={args.dataset}, Model={args.model}, Regulation={args.regulation}, Topk={args.topk}")

        # Call train_model with parsed arguments
        train_model(args.dataset, args.model, args.regulation, args.topk)

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
