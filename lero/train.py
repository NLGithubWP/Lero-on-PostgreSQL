import argparse
import math
import json
import os

from feature import *
from model import LeroModel, LeroModelPairWise

def _load_pointwise_plans(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def _load_pairwise_plans(path):
    X1, X2 = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.split("#####")
            x1, x2 = get_training_pair(arr)
            X1 += x1
            X2 += x2
    return X1, X2


def get_training_pair(candidates):
    assert len(candidates) >= 2
    X1, X2 = [], []

    i = 0
    while i < len(candidates) - 1:
        s1 = candidates[i]
        j = i + 1
        while j < len(candidates):
            s2 = candidates[j]
            X1.append(s1)
            X2.append(s2)
            j += 1
        i += 1
    return X1, X2


def load_training_data(data_dir):
    """Load training data from the specified directory"""
    training_file = os.path.join(data_dir, "training_data.txt")
    X1, X2 = _load_pairwise_plans(training_file)
    
    # Extract execution times as labels
    Y1 = [json.loads(x)[0]['Execution Time'] for x in X1]
    Y2 = [json.loads(x)[0]['Execution Time'] for x in X2]
    
    return X1, X2, Y1, Y2


def compute_rank_score(path, pretrain=False, rank_score_type=0):
    X, Y = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.split("#####")
            if pretrain:
                arr = [(json.loads(p)[0]['Plan']['Total Cost'], p)
                       for p in arr]
            else:
                arr = [(json.loads(p)[0]['Execution Time'], p) for p in arr]
            sorted_arr = sorted(arr, key=lambda x: x[0])

            for i in range(len(sorted_arr)):
                X.append(sorted_arr[i][1])
                if rank_score_type == 0:
                    # 1. x^2
                    print("X^2")
                    Y.append(float((i + 1) ** 2))
                elif rank_score_type == 1:
                    # 2. x^4
                    print("X^4")
                    Y.append(float((i + 1) ** 4))
                elif rank_score_type == 2:
                    # 3. e^x
                    print("e^X")
                    Y.append(float(math.exp(i+1)))
                elif rank_score_type == 3:
                    # 3. x^1
                    print("X^1")
                    Y.append(float((i + 1)))
    return X, Y


def training_pairwise(X1, X2, Y1, Y2, model, pre_training=False, history_file=None):
    """Train the model using pairwise ranking approach"""
    history = model.fit(X1, X2, Y1, Y2, pre_training)
    
    # Save training history to file
    if history_file:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_file}")
    
    return history


def training_with_rank_score(tuning_model_path, model_name, training_data_file, pretrain=False, rank_score_type=0):
    X, Y = compute_rank_score(training_data_file, pretrain, rank_score_type)

    tuning_model = tuning_model_path is not None
    lero_model = None
    if tuning_model:
        lero_model = LeroModel(None)
        lero_model.load(tuning_model_path)
        feature_generator = lero_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X)

    # replace lantency with rank score
    local_features, _ = feature_generator.transform(X)
    assert len(local_features) == len(Y)
    print("Training data set size = " + str(len(local_features)))

    if not tuning_model:
        assert lero_model == None
        lero_model = LeroModel(feature_generator)

    history = lero_model.fit(local_features, Y, tuning_model)

    print(f"saving model... {model_name}")
    lero_model.save(model_name)

    return history


def training_pointwise(tuning_model_path, model_name, training_data_file):
    X = _load_pointwise_plans(training_data_file)

    tuning_model = tuning_model_path is not None
    lero_model = None
    if tuning_model:
        lero_model = LeroModel(None)
        lero_model.load(tuning_model_path)
        feature_generator = lero_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X)

    local_features, y = feature_generator.transform(X)
    assert len(local_features) == len(y)
    print("Training data set size = " + str(len(local_features)))

    if not tuning_model:
        assert lero_model == None
        lero_model = LeroModel(feature_generator)

    history = lero_model.fit(local_features, y, tuning_model)

    print(f"saving model... {model_name}")
    lero_model.save(model_name)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save trained model')
    parser.add_argument('--history_file', type=str,
                      help='File to save training history')
    parser.add_argument('--pre_training', action='store_true',
                      help='Whether this is pre-training phase')
    args = parser.parse_args()

    # Load training data
    X1, X2, Y1, Y2 = load_training_data(args.data_dir)
    
    # Initialize feature generator and model
    feature_generator = FeatureGenerator()
    feature_generator.fit(X1 + X2)
    model = LeroModelPairWise(feature_generator)
    
    # Train model
    history = training_pairwise(X1, X2, Y1, Y2, model, args.pre_training, args.history_file)
    
    # Save model
    model.save(args.model_dir)
    print(f"Model saved to {args.model_dir}")
    
    # Print final metrics
    final_metrics = history[-1]
    print("\nFinal Training Metrics:")
    print(f"Epoch: {final_metrics['epoch']}")
    print(f"Iteration: {final_metrics['iteration']}")
    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
