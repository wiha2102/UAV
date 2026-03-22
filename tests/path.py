"""
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from data.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel
from tests.utils.parsing import CommandSpec, build_parser, mainrunner
from tests.utils.timing import Timer
from src.cfg.data import DataConfig

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)



def print_model_summary(model):
    """Print model architecture summary"""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    
    if hasattr(model, 'encoder'):
        print("\nEncoder:")
        model.encoder.summary()
    
    if hasattr(model, 'decoder'):
        print("\nDecoder:")
        model.decoder.summary()
    
    if hasattr(model, 'vae'):
        print("\nVAE:")
        model.vae.summary()



def print_reconstruction_metrics(y_true, y_pred):
    """Print reconstruction error metrics"""
    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS")
    print("=" * 60)
    
    # Split into components
    n_paths = y_true.shape[1] // 6  # pl(1) + dly(1) + ang(4) = 6 components per path
    
    pl_true = y_true[:, :n_paths]
    pl_pred = y_pred[:, :n_paths]
    ang_true = y_true[:, n_paths:5*n_paths]
    ang_pred = y_pred[:, n_paths:5*n_paths]
    dly_true = y_true[:, 5*n_paths:]
    dly_pred = y_pred[:, 5*n_paths:]
    
    # Path loss error (only for valid paths where pl < max_path_loss)
    # max_path_loss 120-140 dB
    valid_mask = pl_true < 120  # Threshold for valid paths
    pl_rmse = np.sqrt(np.mean((pl_true[valid_mask] - pl_pred[valid_mask])**2))
    
    # Angle error (in degrees)
    ang_error = np.abs(ang_true - ang_pred) * 180  # Since we scaled by 180 in transform
    ang_mae = np.mean(ang_error[valid_mask])
    
    dly_error = np.abs(dly_true - dly_pred)
    dly_mae = np.mean(dly_error[valid_mask])
    
    print(f"Path Loss RMSE: {pl_rmse:.2f} dB")
    print(f"Angle MAE: {ang_mae:.2f} degrees")
    print(f"Delay MAE: {dly_mae:.2e} seconds")


def print_latent_stats(model, x, c):
    """Print latent space statistics"""
    print("\n" + "=" * 60)
    print("LATENT SPACE STATISTICS")
    print("=" * 60)
    
    if hasattr(model, 'path_mod') and hasattr(model.path_mod, 'encoder'):
        z_mu, z_log = model.path_mod.encoder.predict([x, c], verbose=0)
        z = np.random.normal(z_mu, np.exp(0.5 * z_log))
        
        print(f"Latent mean: {z_mu.mean():.4f} ± {z_mu.std():.4f}")
        print(f"Latent variance: {np.exp(z_log).mean():.4f} ± {np.exp(z_log).std():.4f}")
        print(f"Latent dimension: {z.shape[1]}")
        print(f"KL divergence (should be ~0): {(-0.5 * np.mean(1 + z_log - z_mu**2 - np.exp(z_log))):.4f}")

# ============================================================
#       Testing Methods
# ============================================================


def test_build_model(args: argparse.Namespace):
    c = ChannelModel(directory='test')
    c.path.build()
    c.path.model.summary()



def test_train_model(args: argparse.Namespace):
    loader = DataLoader()
    data = loader.load(args.dataset)
    dtr, dts = shuffle_and_split(data, val_ratio=args.ratio)

    model = ChannelModel(directory=args.dataset.split("/")[0])
    model.path.build()

    # Filter to valid links (LOS/NLOS only)
    # dtr = {k: v[dtr['link_state'] != 0] for k, v in dtr.items()}
    # dts = {k: v[dts['link_state'] != 0] for k, v in dts.items()}

    model.path.fit(dtr=dtr, dts=dts, epochs=args.epochs, batch_size=args.batch)
    model.path.save()



def test_evaluate_model(args: argparse.Namespace):
    """Test evaluating the path model"""
    print("\n" + "=" * 60)
    print("TEST: EVALUATE PATH MODEL")
    print("=" * 60)
    
    loader = DataLoader()
    _, dts = shuffle_and_split(loader.load(args.dataset), val_ratio=args.ratio)
    model = ChannelModel(directory=args.dataset.split("/")[0], model_type='vae')
    model.path.load()
    
    # Filter to valid links
    valid_ts = {k: v[dts['link_state'] != 0] for k, v in dts.items()}
    if len(valid_ts['link_state']) == 0:
        print("No valid links in test set!")
        return
    
    print(f"Evaluation samples: {len(valid_ts['link_state'])}")
    
    dvec = valid_ts['dvec']
    rx_type = valid_ts['rx_type']
    los = (valid_ts['link_state'] == 2).astype(np.float32)  # LOS = 2
    
    conditions = model.path._transform_conditions(dvec, rx_type, los, fit=False)    
    features = model.path._transform_data(
        dvec, valid_ts['nlos_pl'], valid_ts['nlos_ang'], valid_ts['nlos_dly'],
        fit=False
    )
    
    print("\nGenerating samples...")
    with Timer("Generation"):
        # Sample from latent space
        if hasattr(model.path.model, 'sample'):
            generated = model.path.model.sample(conditions, n_samples=len(dvec))
        
        else:
            # Fallback: use forward pass with random latent
            z = np.random.normal(0, 1, (len(dvec), model.path.cfg.n_latent))
            generated, _ = model.path.model.decoder.predict([z, conditions], verbose=0)
    
    # Inverse transform to physical domain
    nlos_pl, nlos_ang, nlos_dly = model.path._inverse_transform_data(dvec, generated)
    
    print("\nReconstruction Quality:")
    print(f"  Path Loss RMSE: {np.sqrt(np.mean((valid_ts['nlos_pl'][valid_ts['nlos_pl'] < 120] - nlos_pl[valid_ts['nlos_pl'] < 120])**2)):.2f} dB")
    
    # Angle error (valid paths only)
    valid_paths = valid_ts['nlos_pl'] < 120
    if np.any(valid_paths):
        ang_error = np.abs(valid_ts['nlos_ang'][valid_paths] - nlos_ang[valid_paths]) * 180
        print(f"  Angle MAE: {np.mean(ang_error):.2f} degrees")
        
        dly_error = np.abs(valid_ts['nlos_dly'][valid_paths] - nlos_dly[valid_paths])
        print(f"  Delay MAE: {np.mean(dly_error):.2e} seconds")
    
    print_latent_stats(model, features, conditions)
    if hasattr(model.path.model, 'beta'):
        print(f"\nBeta value: {model.path.model.beta.numpy():.4f}")



def test_sampling_vs_truth(args: argparse.Namespace):
    """Compare generated samples against ground truth distribution"""
    print("\n" + "=" * 60)
    print("TEST: SAMPLE DISTRIBUTION COMPARISON")
    print("=" * 60)
    
    loader = DataLoader()
    _, dts = shuffle_and_split(loader.load(args.dataset), val_ratio=args.ratio)
    model = ChannelModel(directory=args.dataset.split("/")[0], model_type='vae')
    model.path.load()
    
    valid_ts = {k: v[dts['link_state'] != 0] for k, v in dts.items()}
    if len(valid_ts['link_state']) == 0:
        print("No valid links in test set!")
        return
    
    n_samples_per_cond = args.n_samples
    dvec = valid_ts['dvec']
    rx_type = valid_ts['rx_type']
    los = (valid_ts['link_state'] == 2).astype(np.float32)
    
    conditions = model.path._transform_conditions(
        dvec, rx_type, los, fit=False
    )
    
    all_generated = []
    for _ in range(n_samples_per_cond):
        if hasattr(model.path.model, 'sample'):
            generated = model.path.model.sample(conditions)
        
        else:
            z = np.random.normal(0, 1, (len(dvec), model.path.cfg.n_latent))
            generated, _ = model.path.model.decoder.predict([z, conditions], verbose=0)
       
        all_generated.append(generated)
    
    avg_generated = np.mean(all_generated, axis=0)
    features = model.path._transform_data(
        dvec, valid_ts['nlos_pl'], valid_ts['nlos_ang'], valid_ts['nlos_dly'],
        fit=False
    )
    
    # Compare distributions
    print("\nDistribution Statistics:")
    n_paths = model.path.n_max_paths
    for i in range(min(3, n_paths)):  # First 3 paths
        print(f"\nPath {i+1}:")
        print(f"  Ground Truth - Mean: {features[:, i].mean():.4f}, Std: {features[:, i].std():.4f}")
        print(f"  Generated    - Mean: {avg_generated[:, i].mean():.4f}, Std: {avg_generated[:, i].std():.4f}")
        
        eps = 1e-8
        kl = np.mean(features[:, i] * np.log((features[:, i] + eps) / (avg_generated[:, i] + eps)))
        print(f"  KL Divergence: {kl:.4f}")



# ============================================================
#       Mainrunner
# ============================================================

DATA = [
    {"flags": ["--dataset"], "kwargs": {"type": str, "default": "uav_london/train.csv"}},
    {"flags": ["--ratio", "-r"], "kwargs": {"type": float, "default": 0.20}}
]

TRAIN = [
    {"flags": ["--epochs", "-e"], "kwargs": {"type": int, "default": 50}},
    {"flags": ["--batch", "-b"], "kwargs": {"type": int, "default": 256}},
    {"flags": ["--lr"], "kwargs": {"type": float, "default": 1e-4}}
]

EVAL = [
    {"flags": ["--batch", "-b"], "kwargs": {"type": int, "default": 256}},
]

SAMPLE = [
    {"flags": ["--n_samples"], "kwargs": {"type": int, "default": 10}},
]


@mainrunner
def main():
    p = build_parser([
        CommandSpec("build", "Test building path model", test_build_model, []),
        CommandSpec("train", "Test training path model", test_train_model, [*DATA, *TRAIN]),
        CommandSpec("eval", "Test evaluating path model", test_evaluate_model, [*DATA, *EVAL]),
        CommandSpec("sample", "Compare sampling vs ground truth", test_sampling_vs_truth, [*DATA, *SAMPLE]),
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
