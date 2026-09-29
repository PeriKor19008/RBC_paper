import os
import re
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import your custom tools to prevent Training-Serving Skew!
from src.model.experiments.tests_helper import Simulate16BitCamera, get_or_compute_global_max
from src.utils.paths import rel_to_root


def parse_fortran_matrix(file_path):
    """Parses the raw .f06 phase matrix into a 50x50 numpy array."""
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            matches = re.findall(r'[-+]?\d*\.\d+[dD][-+]?\d+', line)
            for match in matches:
                values.append(float(match.replace('D', 'E').replace('d', 'e')))
    return np.array(values).reshape(50, 50)


def predict_single_image(image_path: str | Path, bundle_dir: str | Path, device=None):
    """
    Dynamically loads an ensemble bundle and predicts a single .f06 cell image.
    Returns: (predictions_dict, original_matrix, simulated_matrix)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle_dir = Path(bundle_dir).resolve()
    image_path = Path(image_path).resolve()

    # ---------------------------------------------------------
    # 1. LOAD THE ENSEMBLE ARCHITECTURE
    # ---------------------------------------------------------
    meta_model_path = bundle_dir / "ensemble_meta_learner.pkl"
    if not meta_model_path.exists():
        raise FileNotFoundError(f"Meta-learner not found at {meta_model_path}")
    meta_model = joblib.load(meta_model_path)

    order_txt_path = bundle_dir / "model_order.txt"
    with open(order_txt_path, "r") as f:
        model_filenames = [line.strip() for line in f.readlines() if line.strip()]

    models = []
    for fn in model_filenames:
        model_path = bundle_dir / fn
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)

    # ---------------------------------------------------------
    # 2. PREPARE & SIMULATE CAMERA INPUT
    # ---------------------------------------------------------
    original_matrix = parse_fortran_matrix(image_path)

    # Convert pure matrix to PyTorch Tensor [1, 50, 50]
    raw_tensor = torch.tensor(original_matrix, dtype=torch.float32).unsqueeze(0)

    # Instantiate your exact camera simulator
    global_max = get_or_compute_global_max()
    camera_sim = Simulate16BitCamera(global_max=global_max, burn=True, burn_opt=1)

    # Apply the simulation (16-bit quantization and pixel burning)
    simulated_tensor = camera_sim(raw_tensor)

    # Extract the simulated matrix as a numpy array for plotting
    simulated_matrix = simulated_tensor.squeeze().cpu().numpy()

    # Add the batch dimension -> [1, 1, 50, 50] and push to device
    x_tensor = simulated_tensor.unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # 3. RUN INFERENCE
    # ---------------------------------------------------------
    image_features = []
    with torch.no_grad():
        for m in models:
            pred = m(x_tensor).view(-1).cpu().numpy()
            image_features.extend(pred)

    # Pass combined features to Meta-Learner
    X_input = np.array([image_features])
    final_preds = meta_model.predict(X_input)[0]

    # ---------------------------------------------------------
    # 4. FORMAT RESULTS
    # ---------------------------------------------------------
    results = {
        "Diameter": round(float(final_preds[0]), 3),
        "Thickness": round(float(final_preds[1]), 3),
        "Ratio": round(float(final_preds[2]), 3),
        "Refractive_Index": round(float(final_preds[3])+1000, 3)
    }

    return results, original_matrix, simulated_matrix


def plot_dashboard(filename, results, original_matrix, simulated_matrix):
    """Creates a single-window dashboard showing both states and the predictions."""

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    # Add a main title for the whole window
    fig.suptitle(f"Inference Dashboard: {filename}", fontsize=16, fontweight='bold')

    # Subplot 1: Original Unburnt Image
    ax1 = fig.add_subplot(1, 2, 1)
    c1 = ax1.imshow(original_matrix, cmap='viridis', origin='lower')
    ax1.set_title("Original Raw Phase Matrix", fontsize=12)
    fig.colorbar(c1, ax=ax1, shrink=0.7)

    # Subplot 2: Simulated Burnt Image
    ax2 = fig.add_subplot(1, 2, 2)
    c2 = ax2.imshow(simulated_matrix, cmap='viridis', origin='lower')
    ax2.set_title("Camera Simulated (Burnt & 16-bit)", fontsize=12)
    fig.colorbar(c2, ax=ax2, shrink=0.7)

    # Format the predictions into a clean string
    pred_text = (
        f"Ensemble Predictions\n"
        f"----------------------------------------\n"
        f"Diameter: {results['Diameter']} μm   |   T_max: {results['Thickness']} μm\n"
        f"Ratio: {results['Ratio']}          |   Ref. Index: {results['Refractive_Index']}"
    )

    # Place the text box at the bottom of the window
    fig.text(0.5, 0.05, pred_text, ha='center', va='bottom', fontsize=12,
             fontfamily='monospace',
             bbox=dict(facecolor='#f4f4f4', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Adjust layout so the text doesn't overlap the images
    plt.subplots_adjust(bottom=0.25)

    # Show the interactive window
    plt.show()


# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":

    target_img = rel_to_root("Data/test_data/res_problematic/mesh4x_ref_not_finished/00011a.f06")
    bundle_location = rel_to_root("outputs/models/Ensemble/Ensemble_Bundle_2")

    try:
        # 1. Run the prediction and grab the matrices
        results, original_matrix, simulated_matrix = predict_single_image(target_img, bundle_location)

        # 2. Print to console just in case
        print("\n--- Meta-Learner Ensemble Predictions ---")
        for key, val in results.items():
            print(f"{key:>16}: {val}")

        # 3. Launch the visual dashboard
        file_name = Path(target_img).name
        plot_dashboard(file_name, results, original_matrix, simulated_matrix)

    except Exception as e:
        print(f"Failed to process image: {e}")