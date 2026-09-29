from __future__ import annotations
from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
from src.model.noise import *
from src.utils.norm_cam_transform import *
import re
import pandas as pd
import seaborn as sns
import os
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]



def load_new_format(f, labels_df):
    """
    Loads a 50x50 text-based .f06 image and matches it to its corresponding row in the labels DataFrame.
    """
    # ==========================================
    # 1. --- LOAD THE IMAGE ---
    # ==========================================
    # Read the text file, ignoring empty lines
    lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]

    if len(lines) != 2500:
        raise ValueError(f"Expected 2500 lines for a 50x50 image, got {len(lines)} in: {f.name}")

    # Convert Fortran 'D' notation to standard 'E' notation and cast to floats
    vals = [float(s.replace("D", "E")) for s in lines]

    # Reshape the 1D list into a 2D 50x50 numpy array
    arr = np.asarray(vals, dtype=np.float32).reshape(50, 50)

    # Convert to PyTorch tensor and add the channel dimension -> shape: [1, 50, 50]
    img = torch.from_numpy(arr).unsqueeze(0)

    # ==========================================
    # 2. --- FIND THE MATCHING LABEL ---
    # ==========================================
    # Find the block of numbers in the filename (e.g., "09171" from "09171a.f06")
    match = re.search(r'(\d+)', f.name)

    if not match:
        raise ValueError(f"Could not find an index number in filename: {f.name}")

    # Get the full string of digits
    full_digits = match.group(1)

    # Slice off the very last digit (e.g., "09171" becomes "0917")
    actual_number = full_digits[:-1]

    # Convert to integer and subtract 1 for the 0-based Pandas index
    img_idx = int(actual_number) - 1

    # Paranoia Check: Make sure the row actually exists!
    if img_idx >= len(labels_df):
        raise IndexError(f"\n[CRASH AVERTED] Mismatch detected!\n"
                         f"Image '{f.name}' translated to Index {img_idx}, "
                         f"but 'clean_labels.txt' only contains {len(labels_df)} rows.")

    # Extract that specific row
    row = labels_df.iloc[img_idx]

    # ==========================================
    # 3. --- CONVERT LABEL TO TENSOR ---
    # ==========================================
    lbl_true = torch.tensor([
        row['d'] * 1000.0,  # Scale to nanometers
        row['tmax'] * 1000.0,  # Scale to nanometers
        row['thick_ratio'] * 1000.0,  # Scale up by 1000
        (row['ref_index'] - 1.0) * 1000.0  # Discard the "1." and scale up
    ], dtype=torch.float32)

    return img, lbl_true




def change_block(size, img:Tensor) -> Tensor:
    if size < 1:
        raise ValueError("size must be >= 1")


    out = img.clone()

    _, H, W = out.shape
    b = int(size)
    b = min(b, H, W)

    # choose top-left corner uniformly
    i = torch.randint(0, H - b + 1, (1,)).item()
    j = torch.randint(0, W - b + 1, (1,)).item()

    # fill with random values sampled uniformly between image min/max
    low = float(out.min())
    high = float(out.max())
    patch = torch.empty((1, b, b), dtype=out.dtype, device=out.device).uniform_(low, high)

    out[:, i:i + b, j:j + b] = patch
    return out


def jitter_block(size: int, img: Tensor, strength: float = 0.1) -> Tensor:

    if size < 1:
        raise ValueError("size must be >= 1")

    out = img.clone()
    _, H, W = out.shape
    b = int(size)
    b = min(b, H, W)

    # choose top-left corner
    i = torch.randint(0, H - b + 1, (1,)).item()
    j = torch.randint(0, W - b + 1, (1,)).item()

    # current block and its local stats
    block = out[:, i:i+b, j:j+b]
    local_std = float(block.std())
    global_std = float(out.std())
    sigma = strength * (local_std if local_std > 0.0 else global_std)

    if sigma > 0.0:
        noise = torch.randn_like(block) * sigma
        block = block + noise  # small random change
    # else sigma==0 -> block is constant; leave it as-is

    print(sigma)
    low = float(out.min())
    high = float(out.max())
    block = block.clamp(min=low, max=high)

    out[:, i:i+b, j:j+b] = block
    return out

def show_img(img: torch.Tensor):
    arr = img.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    arr = arr.numpy()

    vmin, vmax = float(arr.min()), float(arr.max())

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = plt.gca()
    im = ax.imshow(
        arr,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )


    fig.tight_layout()


    plt.show()
    plt.close(fig)


def plot_error_prc(iterations, errors: List[float], max_errors: List[float], std_errors: List[float] = None,
                   save_path: str | None = None, show: bool = False):
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    avg_vals = [float(v) for v in errors]
    std_vals = [float(v) for v in std_errors] if std_errors else None

    # --- NEW CODE: Create custom x-axis labels with Std Dev ---
    if std_vals:
        x_labels = [f"{key}\n(±{std:.2f}%)" for key, std in zip(LABEL_KEYS, std_vals)]
    else:
        x_labels = LABEL_KEYS
    # ----------------------------------------------------------

    x = np.arange(len(LABEL_KEYS))
    width_single = 0.6
    width_grouped = 0.38

    # Slightly increased the height from 4.5 to 5.0 to give space for the two-line x-labels
    fig = plt.figure(figsize=(7.5, 5.0))
    ax = plt.gca()

    if max_errors is None:
        ax.set_ylim(0, max(avg_vals) * 1.15)
        bars = ax.bar(x, avg_vals, width_single, label="Avg |Error|")
        title = f"Average Error Percentage across {iterations} samples"

        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3g}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        ax.legend()
    else:
        max_vals = [float(v) for v in max_errors]

        ymax = max(max(avg_vals), max(max_vals)) * 1.15
        ax.set_ylim(0, ymax)

        bars_avg = ax.bar(x - width_grouped / 2, avg_vals, width_grouped, label="Avg |Error|")
        bars_max = ax.bar(x + width_grouped / 2, max_vals, width_grouped, label="Max |Error|")

        title = f"Avg & Max Error Percentage across {iterations} samples"

        for b in list(bars_avg) + list(bars_max):
            h = b.get_height()
            ax.annotate(f"{h:.3g}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        ax.legend()

    ax.set_xticks(x)
    # --- FIXED: Use the custom labels here ---
    ax.set_xticklabels(x_labels)
    # -----------------------------------------
    ax.set_ylabel("Error Percentage")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def test_single_label(model: nn.Module, dir_path: str | Path, save_path_pct: str | None = None, thresh: float = 15.0,
                      block: bool = False, jitter: bool = False, noise: bool = True, normalize: bool = False, target_idx: int = 0,
                      ae: nn.Module = None):  # <--- MODIFIED: Added target_idx parameter

    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    model.eval()
    dev = next(model.parameters()).device

    # For nice printing
    label_names = ["diameter", "thickness", "ratio", "ref_index"]
    target_name = label_names[target_idx]
    print(f"\n=== Testing Specialized CNN for: {target_name.upper()} ===")

    # MODIFIED: Changed accumulators from size 4 arrays to single values
    error = 0.0
    error_prc = 0.0
    max_prc_err = 0.0
    it = 0
    all_prc_errors = []
    all_y_true = []
    all_y_pred = []

    labels_file_path = dir_path.parent / "clean_labels.txt"
    all_labels_df = pd.read_csv(labels_file_path)  # read_csv reads comma-separated txt files perfectly!
    print("\n[DEBUG] Pandas found these columns:", all_labels_df.columns.tolist())
    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue

        # --- NEW: Pass the dataframe into our new loader ---
        img, lbl_true = load_new_format(f, all_labels_df)

        if normalize == 1:
            sim = Simulate16BitCamera(get_or_compute_global_max(), burn_opt=1)
            img = sim(img)


        if block:
            img = change_block(2, img)
        if jitter:
            img = jitter_block(5, img, 5)

        if noise:
            n = nn.Sequential(
                AddGaussianNoise(std=0.8, p=0.5),
                AddSpeckleNoise(std=0.8, p=0.5),
            )
            img = n(img)

        x = img.unsqueeze(0).to(dev)

        if ae:
            x = ae(x)

        with torch.no_grad():
            # MODIFIED: The model now outputs 1 scalar value
            lbl_pred_single = model(x).squeeze().detach().cpu()

        # MODIFIED: Extract the 1 correct true label from the 4-label array
        lbl_true_single = lbl_true[target_idx].cpu()

        # Calculate error for this single label
        abs_err = abs(lbl_true_single - lbl_pred_single).item()
        eps = 1e-8
        prc_err = (abs_err / (abs(lbl_true_single.item()) + eps)) * 100.0

        if prc_err > thresh:
            print(f"\n[!] Outlier Detected: {f.name}")
            print(f"    {'Property':<12} | {'True Value':<12} | {'Predicted':<12} | {'Error %':<10}")
            print("    " + "-" * 55)
            print(
                f"    {target_name:<12} | {lbl_true_single.item():<12.5g} | {lbl_pred_single.item():<12.5g} | {prc_err:>6.2f}%  <-- OVER THRESHOLD")
            print("    " + "-" * 55 + "\n")

        # <--- CRITICAL FIX: Removed the 'else:' block. We must accumulate errors for ALL samples!
        error += abs_err
        error_prc += prc_err
        if prc_err > max_prc_err:
            max_prc_err = prc_err
        it += 1

        err_padded = [0.0, 0.0, 0.0, 0.0]
        err_padded[target_idx] = prc_err
        all_prc_errors.append(err_padded)

        true_padded = [0.0, 0.0, 0.0, 0.0]
        true_padded[target_idx] = lbl_true_single.item()
        all_y_true.append(true_padded)

        pred_padded = [0.0, 0.0, 0.0, 0.0]
        pred_padded[target_idx] = lbl_pred_single.item()
        all_y_pred.append(pred_padded)

    # MODIFIED: Averages for plotting (Wrapped in lists to mimic the old 4-label format for the plot scripts)
    avg_prc_err = [0.0, 0.0, 0.0, 0.0]
    avg_prc_err[target_idx] = error_prc / it

    max_prc_err_list = [0.0, 0.0, 0.0, 0.0]
    max_prc_err_list[target_idx] = max_prc_err

    errors_tensor = torch.tensor(all_prc_errors, dtype=torch.float32)
    std_prc_err = torch.std(errors_tensor, dim=0).tolist()
    errors_tensor = torch.tensor(all_prc_errors, dtype=torch.float32)
    std_prc_err = torch.std(errors_tensor, dim=0).tolist()

    plot_error_prc(it, avg_prc_err, max_prc_err_list, std_prc_err, str(save_path_pct))

    y_true_np = np.array(all_y_true)
    y_pred_np = np.array(all_y_pred)
    y_true_ts = torch.tensor(all_y_true, dtype=torch.float32)
    y_pred_ts = torch.tensor(all_y_pred, dtype=torch.float32)

    # Define output directory
    dir_out = os.path.dirname(str(save_path_pct))

    # 1. R-Squared Scores
    print_r2_scores(y_true_ts, y_pred_ts)

    # 2. Boxplot
    plot_boxplot_errors(all_prc_errors, save_path=os.path.join(dir_out, "boxplot_errors.png"))

    # 3. Scatter Plot
    plot_scatter_true_vs_pred(y_true_np, y_pred_np, save_path=os.path.join(dir_out, "scatter_true_pred.png"))

    # 4. Bland-Altman
    plot_bland_altman(y_true_np, y_pred_np, save_path=os.path.join(dir_out, "bland_altman.png"))
    # -------------------------------------------------

    print("######")
    print(f"Results for {target_name.upper()}:")
    print(" avg error ------ " + str(avg_prc_err[0]))
    print(" max error ------ " + str(max_prc_err_list[0]))
    print(" std error ------ " + str(std_prc_err[0]))


def test_single_label_ensemble(models: list, dir_path: str | Path, save_path_pct: str | None = None,
                               thresh: float = 15.0,
                               block: bool = False, jitter: bool = False, noise: bool = True, normalize: bool = False,
                               ae: nn.Module = None, trimed: bool = False, target_idx: int = 0, min_val: float | None = None,):
    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Ensure all models are in eval mode and grab the device
    for m in models:
        m.eval()
    dev = next(models[0].parameters()).device

    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    target_name = LABEL_KEYS[target_idx]
    print(f"\n=== Testing Ensemble ({len(models)} models) for: {target_name.upper()} ===")

    # Accumulators for the single label
    error = 0.0
    error_prc = 0.0
    max_prc_err = 0.0
    it = 1
    all_prc_errors = []
    all_y_true = []
    all_y_pred = []

    labels_file_path = dir_path.parent / "clean_labels.txt"
    all_labels_df = pd.read_csv(labels_file_path)  # read_csv reads comma-separated txt files perfectly!
    print("\n[DEBUG] Pandas found these columns:", all_labels_df.columns.tolist())
    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue


        img, lbl_true = load_new_format(f, all_labels_df)
        lbl_true_single = lbl_true[target_idx].cpu()


        if min_val is not None and lbl_true_single.item() < min_val:
            continue
        if normalize == 1:
            sim = Simulate16BitCamera(get_or_compute_global_max(), burn_opt=1)
            img = sim(img)

        if block:
            img = change_block(2, img)
        if jitter:
            img = jitter_block(5, img, 5)

        if noise:
            n = nn.Sequential(
                AddGaussianNoise(std=0.8, p=0.5),
                AddSpeckleNoise(std=0.8, p=0.5),
            )
            img = n(img)

        x = img.unsqueeze(0).to(dev)

        if ae:
            x = ae(x)

        with torch.no_grad():
            model_preds = []
            for m in models:
                # Get single prediction from each model
                pred = m(x).squeeze().detach().cpu()
                model_preds.append(pred)

            # Stack into a 1D tensor of shape [Num_Models]
            stacked_preds = torch.stack(model_preds)

            # We need at least 3 models to drop min/max and still have one left!
            if trimed and len(models) > 2:
                # Sort the predictions
                sorted_preds, _ = torch.sort(stacked_preds, dim=0)

                # Slice off the min (first) and max (last)
                trimmed_preds = sorted_preds[1:-1]

                # Calculate the average of the remaining "safe" predictions
                lbl_pred_single = torch.mean(trimmed_preds, dim=0)
            else:
                # Fallback to standard average
                lbl_pred_single = torch.mean(stacked_preds, dim=0)

        # Extract the 1 correct true label
        lbl_true_single = lbl_true[target_idx].cpu()

        # Calculate error for this single label
        abs_err = abs(lbl_true_single - lbl_pred_single).item()
        eps = 1e-8
        prc_err = (abs_err / (abs(lbl_true_single.item()) + eps)) * 100.0

        if prc_err > thresh:
            print(f"\n[!] Outlier Detected: {f.name}")
            print(f"    {'Property':<12} | {'True Value':<12} | {'Predicted':<12} | {'Error %':<10}")
            print("    " + "-" * 55)
            print(
                f"    {target_name:<12} | {lbl_true_single.item():<12.5g} | {lbl_pred_single.item():<12.5g} | {prc_err:>6.2f}%  <-- OVER THRESHOLD")
            print("    " + "-" * 55 + "\n")

        error += abs_err
        error_prc += prc_err
        if prc_err > max_prc_err:
            max_prc_err = prc_err
        it += 1

        # ==========================================================
        # PAD WITH ZEROS: Keep Matplotlib happy with 4 columns
        # ==========================================================
        err_padded = [0.0, 0.0, 0.0, 0.0]
        err_padded[target_idx] = prc_err
        all_prc_errors.append(err_padded)

        true_padded = [0.0, 0.0, 0.0, 0.0]
        true_padded[target_idx] = lbl_true_single.item()
        all_y_true.append(true_padded)

        pred_padded = [0.0, 0.0, 0.0, 0.0]
        pred_padded[target_idx] = lbl_pred_single.item()
        all_y_pred.append(pred_padded)

    # Pad the averages
    avg_prc_err = [0.0, 0.0, 0.0, 0.0]
    avg_prc_err[target_idx] = error_prc / it

    max_prc_err_list = [0.0, 0.0, 0.0, 0.0]
    max_prc_err_list[target_idx] = max_prc_err

    errors_tensor = torch.tensor(all_prc_errors, dtype=torch.float32)
    std_prc_err = torch.std(errors_tensor, dim=0).tolist()

    plot_error_prc(it, avg_prc_err, max_prc_err_list, std_prc_err, str(save_path_pct))

    y_true_np = np.array(all_y_true)
    y_pred_np = np.array(all_y_pred)
    y_true_ts = torch.tensor(all_y_true, dtype=torch.float32)
    y_pred_ts = torch.tensor(all_y_pred, dtype=torch.float32)

    # Define output directory
    dir_out = os.path.dirname(str(save_path_pct))

    # 1. R-Squared Scores
    print_r2_scores(y_true_ts, y_pred_ts)

    # 2. Boxplot
    plot_boxplot_errors(all_prc_errors, save_path=os.path.join(dir_out, "boxplot_errors.png"))

    # 3. Scatter Plot
    plot_scatter_true_vs_pred(y_true_np, y_pred_np, save_path=os.path.join(dir_out, "scatter_true_pred.png"))

    # 4. Bland-Altman
    plot_bland_altman(y_true_np, y_pred_np, save_path=os.path.join(dir_out, "bland_altman.png"))

    plot_percentage_residuals(y_true_np, y_pred_np, save_path=os.path.join(dir_out, "percentage_residuals.png"))
    # -------------------------------------------------
    min_true = torch.min(y_true_ts[:, target_idx]).item()
    max_true = torch.max(y_true_ts[:, target_idx]).item()

    min_pred = torch.min(y_pred_ts[:, target_idx]).item()
    max_pred = torch.max(y_pred_ts[:, target_idx]).item()
    print("\n###### ENSEMBLE RESULTS ######")
    print(f"Results for {target_name.upper()}:")
    print(" avg error ------ " + str(avg_prc_err[target_idx]))
    print(" max error ------ " + str(max_prc_err_list[target_idx]))
    print(" std error ------ " + str(std_prc_err[target_idx]))
    print("-" * 30)
    print(f" Min True Value : {min_true:.5g}")
    print(f" Max True Value : {max_true:.5g}")
    print(f" Min Pred Value : {min_pred:.5g}")
    print(f" Max Pred Value : {max_pred:.5g}")
    print("##############################\n")

    return avg_prc_err[target_idx], max_prc_err_list[target_idx]


def plot_scatter_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None, show: bool = False):
    """ Scatter plot of True vs Predicted Values (2x2 grid) using Hexbins for large data """
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Made slightly wider to fit the colorbars comfortably
    fig.suptitle("True vs Predicted Values", fontsize=16)

    for i, ax in enumerate(axs.flat):
        # 1. Extract the 1D arrays for this specific label
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        # 2. Use ax.hexbin and pass the 1D arrays (true_vals, pred_vals)
        # We save it to 'hb' so we can link the colorbar to it
        hb = ax.hexbin(true_vals, pred_vals, gridsize=40, cmap='Blues', mincnt=1)

        # 3. Attach the colorbar directly to this specific subplot
        fig.colorbar(hb, ax=ax, label='Count in bin')

        # Draw the ideal diagonal line (y = x)
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")

        ax.set_title(LABEL_KEYS[i].capitalize(), fontsize=14)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    plt.close(fig)

def plot_percentage_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None, show: bool = False):
    """ Plots True Value vs. Percentage Error to magnify tiny deviations """
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Percentage Error Residuals (Magnified)", fontsize=16)

    for i, ax in enumerate(axs.flat):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        # Calculate the percentage error for every single point
        eps = 1e-8 # Prevent division by zero
        pct_errors = ((pred_vals - true_vals) / (np.abs(true_vals) + eps)) * 100.0

        # Plot using hexbin so it doesn't overplot
        hb = ax.hexbin(true_vals, pct_errors, gridsize=40, cmap='Reds', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Count')

        # Draw a thick black line at 0% error (Perfect Prediction)
        ax.axhline(0, color='black', linewidth=2, linestyle='--', label="0% Error (Ideal)")

        # OPTIONAL: Force the Y-axis to zoom in strictly between -5% and +5% error
        # This acts like a magnifying glass, cutting out crazy outliers
        # ax.set_ylim(-5.0, 5.0)

        ax.set_title(LABEL_KEYS[i].capitalize(), fontsize=14)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Error (%)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_boxplot_errors(all_prc_errors: list, save_path: str | None = None, show: bool = False):
    """ Plots the distribution of percentage errors using a clean Violin Plot """
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Percentage Error Distribution", fontsize=16)

    # --- DATA SANITIZATION STEP ---
    # Force the data into a pure, clean 2D NumPy array of floats.
    # This strips away any PyTorch tensors or weird list structures that confuse Seaborn.
    clean_data = np.asarray(all_prc_errors, dtype=np.float32)

    # If the data accidentally came in sideways (e.g., shape [4, 3500] instead of [3500, 4]),
    # we transpose it so Seaborn knows there are exactly 4 columns.
    if clean_data.shape[0] == len(LABEL_KEYS):
        clean_data = clean_data.T

    # --- THE VIOLIN PLOT ---
    sns.violinplot(
        data=clean_data,
        inner="quartile",
        cut=0,
        palette="muted",
        linewidth=1.5,
        ax=ax
    )

    # Apply your labels to the X-axis
    ax.set_xticks(range(len(LABEL_KEYS)))
    ax.set_xticklabels(LABEL_KEYS, fontsize=12)

    ax.set_ylabel("Error Percentage (%)", fontsize=12)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        # Save the file silently
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)  # Clear the memory


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None, show: bool = False):
    """ Bland-Altman Plot (Mean vs Difference) (2x2 grid) """
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Bland-Altman Plots", fontsize=14)

    for i, ax in enumerate(axs.flat):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        mean_vals = (true_vals + pred_vals) / 2.0
        diff_vals = pred_vals - true_vals  # Error (Predicted - True)

        md = np.mean(diff_vals)  # Mean Difference
        sd = np.std(diff_vals, axis=0)  # Standard Deviation of Difference

        ax.scatter(mean_vals, diff_vals, alpha=0.6, edgecolors='k')
        ax.axhline(md, color='red', linestyle='-', lw=2, label=f'Mean Diff: {md:.2g}')
        ax.axhline(md + 1.96 * sd, color='gray', linestyle='--', lw=2, label='+1.96 SD')
        ax.axhline(md - 1.96 * sd, color='gray', linestyle='--', lw=2, label='-1.96 SD')

        ax.set_title(LABEL_KEYS[i])
        ax.set_xlabel("Mean of True and Pred")
        ax.set_ylabel("Difference (Pred - True)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def print_r2_scores(y_true: torch.Tensor, y_pred: torch.Tensor):
    """ Calculate and print the R-squared (R2) Score """
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]

    # R2 = 1 - ( SS_res / SS_tot )
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)

    r2_scores = 1 - (ss_res / (ss_tot + 1e-8))  # 1e-8 prevents division by zero

    print("\n--- R-squared (R²) Scores ---")
    for i in range(4):
        print(f"{LABEL_KEYS[i]}: {r2_scores[i].item():.4f}  (or {r2_scores[i].item() * 100:.2f}%)")
    print("-----------------------------\n")