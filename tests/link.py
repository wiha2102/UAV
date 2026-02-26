import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from data.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel
from tests.utils.parsing import CommandSpec, build_parser, mainrunner
from tests.utils.timing import Timer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)



def print_confusion_matrix(cm, class_names=None):
    """Pretty print confusion matrix"""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    # Header
    header = "Actual \\ Predicted" + " " * 5
    for name in class_names:
        header += f"{name[:8]:>8}"
    print(header)
    print("-" * 60)
    
    # Rows
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<20}"
        for val in row:
            row_str += f"{val:>8}"
        print(row_str)
    print("=" * 60)


def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed classification metrics"""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Calculate metrics for each class
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        # Binary mask for current class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(y_true == i)
        
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")


def calculate_top_k_accuracy(y_true, y_prob, k=2):
    """Calculate top-k accuracy"""
    # Get indices of top k predictions
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    
    # Check if true label is in top k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


# ============================================================
#       Testing Methods
# ============================================================

def test_build_model(args: argparse.Namespace):
    c = ChannelModel(directory='test')
    c.link.build()
    c.link.model.summary()


def test_train_model(args: argparse.Namespace):
    loader = DataLoader()
    dtr, dts = shuffle_and_split(loader.load(args.dataset),val_ratio=args.ratio)

    model = ChannelModel(directory=args.dataset.split("/")[0])
    model.link.build()
    model.link.fit(dtr=dtr, dts=dts,epochs=args.epochs,batch_size=args.batch)
    model.link.save()


def test_evaluate_model(args: argparse.Namespace):
    loader = DataLoader()
    _, dts = shuffle_and_split(loader.load(args.dataset),val_ratio=args.ratio)

    model = ChannelModel(directory=args.dataset.split("/")[0])
    model.link.load()

    x_test, y_test = model.link._prepare_arrays(dts, fit=False)
    y_prob = model.link.model.predict(x_test, batch_size=args.batch,verbose=1)
    y_pred = np.argmax(y_prob,axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Class names based on LinkState enum
    class_names = ["No-Link", "NLOS", "LOS"]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, class_names)

    # Detailed classification report
    print_classification_report(y_test, y_pred, class_names)
    
    # Top-2 accuracy (useful for 3-class problem)
    top2_acc = calculate_top_k_accuracy(y_test, y_prob, k=2)
    print(f"\nTop-2 Accuracy: {top2_acc:.4f}")
    
    # Class distribution
    print(f"\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for class_id, count, class_name in zip(unique, counts, class_names):
        percentage = count / len(y_test) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    



# ============================================================
#       Mainrunner
# ============================================================

DATA = [
    {"flags":["--dataset"],"kwargs":{"type":str,"default":"uav_london/train.csv"}},
    {"flags":["--ratio","-r"],"kwargs":{"type":float,"default":0.20}}
]
TRAIN=[
    {"flags":["--epochs","-e"],"kwargs":{"type":int,"default":100}},
    {"flags":["--batch","-b"],"kwargs":{"type":int,"default":512}}
]

@mainrunner
def main():
    p = build_parser([
        CommandSpec("build","Test Build",test_build_model,[]),
        CommandSpec("train","Test training",test_train_model,[*DATA,*TRAIN]),
        CommandSpec("eval","Test evaluate",test_evaluate_model,[*DATA,*TRAIN])
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
