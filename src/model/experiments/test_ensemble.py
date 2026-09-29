import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import joblib

# Import your custom functions AND your plotting functions!
from src.model.experiments.tests_helper import *
from src.utils.paths import rel_to_root


def extract_testing_features(models: list, test_dir: str | Path, normalize: bool = False):
    """
    Runs the final holdout images through all base models to create the test spreadsheet.
    """
    test_dir = Path(test_dir).resolve()

    for m in models:
        m.eval()
    dev = next(models[0].parameters()).device

    labels_file_path = test_dir.parent / "clean_labels.txt"
    all_labels_df = pd.read_csv(labels_file_path)

    # Added file_names list to track which image is which
    X_list, y_list, file_names = [], [], []
    files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() == ".f06"]

    print(f"Extracting features from {len(files)} test images in {test_dir.name}...")

    with torch.no_grad():
        for f in files:
            img, lbl_true = load_new_format(f, all_labels_df)
            if normalize:
                sim = Simulate16BitCamera(get_or_compute_global_max(), burn=True, burn_opt=1)
                img = sim(img)

            x_tensor = img.unsqueeze(0).to(dev)

            image_features = []
            for m in models:
                pred = m(x_tensor).view(-1).cpu().numpy()
                image_features.extend(pred)

            X_list.append(image_features)
            y_list.append(lbl_true.cpu().numpy())
            file_names.append(f.name)  # Store the filename

    return np.array(X_list), np.array(y_list), file_names


def grade_ensemble(meta_model, X_test, y_test, file_names, graphs_save_dir: str, thresholds: list = [0, 0, 0, 0],err_threshold=20.0):
    """
    Takes the final exam, applies biological thresholds, prints metrics, and generates graphs.
    """
    print("\n==================================================")
    print(f" THE FINAL EXAM (HOLDOUT SET)")
    print(f" Total Testing Images Evaluated: {X_test.shape[0]}")
    print("==================================================")

    y_pred = meta_model.predict(X_test)
    LABEL_KEYS = ["Diameter", "Thickness", "Ratio", "Refractive Index"]

    # ---------------------------------------------------------
    # 1. APPLY GLOBAL THRESHOLD FILTERING
    # ---------------------------------------------------------
    # Start assuming all images are valid
    global_mask = np.ones(len(y_test), dtype=bool)

    # Update the mask: An image must pass ALL active thresholds to survive
    for i, (min_t, max_t) in enumerate(thresholds):
        # Έλεγχος αν το κατώφλι είναι ενεργό (min_t > 0 ή max_t < np.inf)
        if min_t > 0:
            global_mask &= (y_test[:, i] >= min_t)
        if max_t < np.inf:
            global_mask &= (y_test[:, i] <= max_t)

    filtered_count = len(y_test) - np.sum(global_mask)
    if filtered_count > 0:
        print(f"\n ⚠️ GLOBAL FILTER APPLIED:")
        for i, (min_t, max_t) in enumerate(thresholds):
            # Ελέγχουμε αν υπάρχει ενεργό κατώφλι (min > 0) ή ανώτατο όριο (max < np.inf)
            if min_t > 0 or max_t < np.inf:
                limits = []
                if min_t > 0: limits.append(f">= {min_t}")
                if max_t < np.inf: limits.append(f"<= {max_t}")
                print(f"    - {LABEL_KEYS[i]} ({' & '.join(limits)})")

        print(f"    - Removed {filtered_count} physically invalid edge-case images.")
        print(f"    - Remaining Valid Images: {np.sum(global_mask)} / {len(y_test)}\n")

    # Slice the arrays to strictly keep only the surviving images
    y_test_valid = y_test[global_mask]
    y_pred_valid = y_pred[global_mask]
    valid_filenames = np.array(file_names)[global_mask]

    # Stop execution if filtering removed everything
    if len(y_test_valid) == 0:
        print(" ❌ WARNING: Thresholds filtered out ALL images!")
        return

    # ---------------------------------------------------------
    # 2. CALCULATE METRICS ON SURVIVING DATA
    # ---------------------------------------------------------
    avg_errors_list = []
    max_errors_list = []
    std_errors_list = []
    high_error_report = {}

    # For plotting: we pad the invalid rows with NaN so Seaborn ignores them
    all_prc_errors_matrix = np.full_like(y_test, fill_value=np.nan)
    y_test_plot = y_test.copy()
    y_pred_plot = y_pred.copy()
    y_test_plot[~global_mask] = np.nan
    y_pred_plot[~global_mask] = np.nan

    for idx in range(4):
        print(f"--- {LABEL_KEYS[idx].upper()} METRICS ---")

        true_vals = y_test_valid[:, idx]
        pred_vals = y_pred_valid[:, idx]

        eps = 1e-8
        prc_errors = (np.abs(true_vals - pred_vals) / (np.abs(true_vals) + eps)) * 100.0

        # IDENTIFY ERRORS > err_threshold
        bad_indices = np.where(prc_errors > err_threshold)[0]
        for b_idx in bad_indices:
            fname = valid_filenames[b_idx]

            # If this is the first time this file triggered an alarm, store ALL its parameters
            if fname not in high_error_report:
                all_true = y_test_valid[b_idx]
                all_pred = y_pred_valid[b_idx]

                # Calculate percentage errors for all 4 parameters at once
                all_errs = (np.abs(all_true - all_pred) / (np.abs(all_true) + eps)) * 100.0

                high_error_report[fname] = {
                    'true': all_true,
                    'pred': all_pred,
                    'errors': all_errs
                }

        # Map the surviving errors back to the correct rows in the NaN-padded matrix
        all_prc_errors_matrix[global_mask, idx] = prc_errors

        avg_err = np.mean(prc_errors)
        max_err = np.max(prc_errors)
        std_err = np.std(prc_errors)

        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + eps))

        min_true, max_true = np.min(true_vals), np.max(true_vals)
        min_pred, max_pred = np.min(pred_vals), np.max(pred_vals)

        print(f" Avg Error ------ {avg_err:.3f}%")
        print(f" Max Error ------ {max_err:.3f}%")
        print(f" Std Error ------ {std_err:.3f}%")
        print(f" R² Score  ------ {r2_score:.4f}")
        print(f" Min/Max True   : {min_true:.4g} / {max_true:.4g}")
        print(f" Min/Max Pred   : {min_pred:.4g} / {max_pred:.4g}\n")

        avg_errors_list.append(avg_err)
        max_errors_list.append(max_err)
        std_errors_list.append(std_err)

        # ---------------------------------------------------------
        # PRINT HIGH ERROR REPORT
        # ---------------------------------------------------------
    if high_error_report:
        print("==================================================")
        print(f" 🚨 IMAGES WITH >{err_threshold}% ERROR DETECTED")
        print("==================================================")
        for fname, data in high_error_report.items():
            print(f"File: {fname}")

            # Loop through all 4 features to print full context for this specific image
            for j, label_name in enumerate(LABEL_KEYS):
                err_val = data['errors'][j]
                t_val = data['true'][j]
                p_val = data['pred'][j]

                # Places a cross mark ❌ only next to the specific property that failed
                marker = "❌" if err_val > err_threshold else "  "

                print(f"  {marker} {label_name}: {err_val:.2f}% error (True: {t_val:.4g}, Pred: {p_val:.4g})")
            print("-" * 50)
        print("==================================================\n")
    else:
        print(f" ✅ SUCCESS: Zero images with >{err_threshold}% error in the valid dataset!\n")

    # ---------------------------------------------------------
    # 3. GENERATE ALL THE GRAPHS
    # ---------------------------------------------------------
    print("Generating Publication Graphs...")
    os.makedirs(graphs_save_dir, exist_ok=True)

    plot_scatter_true_vs_pred(y_test_plot, y_pred_plot,
                              save_path=os.path.join(graphs_save_dir, "scatter_true_vs_pred.png"), show=True)
    plot_percentage_residuals(y_test_plot, y_pred_plot,
                              save_path=os.path.join(graphs_save_dir, "residuals_magnified.png"), show=True)
    plot_bland_altman(y_test_plot, y_pred_plot, save_path=os.path.join(graphs_save_dir, "bland_altman.png"), show=True)
    plot_boxplot_errors(all_prc_errors_matrix, save_path=os.path.join(graphs_save_dir, "violin_errors.png"), show=True)
    plot_error_prc(
        iterations=X_test.shape[0], errors=avg_errors_list, max_errors=max_errors_list,
        std_errors=std_errors_list, save_path=os.path.join(graphs_save_dir, "error_bar_chart.png"), show=True
    )
    print(f"✅ All graphs saved securely to: {graphs_save_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    bundle_dir = rel_to_root("outputs/models/Ensemble/Ensemble_Bundle_3")

    # 1. LOAD THE META-LEARNER
    meta_model_path = os.path.join(bundle_dir, "ensemble_meta_learner.pkl")
    meta_model = joblib.load(meta_model_path)

    # 2. READ THE EXACT ORDER FROM THE TEXT FILE
    order_txt_path = os.path.join(bundle_dir, "model_order.txt")
    with open(order_txt_path, "r") as f:
        model_filenames = [line.strip() for line in f.readlines() if line.strip()]

    # 3. SAFELY LOAD THE BUNDLED MODELS
    print(f"Loading {len(model_filenames)} Base Models automatically from manifest...")
    models = [torch.load(os.path.join(bundle_dir, fn), map_location="cpu", weights_only=False).to(device) for fn in
              model_filenames]

    # 4. POINT TO THE FINAL TEST FOLDER
    dir_final_test = rel_to_root("Data/test_data/rs_final_test")

    # 5. DEFINE YOUR THRESHOLDS [Diameter, Thickness, Ratio, Ref_Index]
    # Set to 0 to ignore. Set > 0 to filter out true values below that limit.
    # my_thresholds = [0, 0, 450, 15]
    thresholds: list = [(5800, 10000), (1700, 2900), (0, np.inf), (15, np.inf)]

    # 6. RUN EXTRACTION
    X_test, y_test, file_names = extract_testing_features(models, dir_final_test, normalize=True)

    # 7. GRADE & GRAPH
    graphs_folder = os.path.join(bundle_dir, "plots")
    grade_ensemble(meta_model, X_test, y_test, file_names, graphs_save_dir=graphs_folder, thresholds=thresholds,err_threshold=5.0)



