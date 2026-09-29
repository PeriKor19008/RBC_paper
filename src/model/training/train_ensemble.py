import os
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib
from src.model.experiments.tests_helper import *
from src.utils.paths import rel_to_root


# ==========================================
# IMPORTANT: Import your custom functions here!
# from your_module import load_new_format, Simulate16BitCamera, get_or_compute_global_max, rel_to_root
# ==========================================

def extract_training_features(models: list, train_dir: str | Path, normalize: bool = False):
    """
    Runs training images through all base models to create a flat dataset of predictions.
    """
    train_dir = Path(train_dir).resolve()

    for m in models:
        m.eval()
    dev = next(models[0].parameters()).device

    labels_file_path = train_dir.parent / "clean_labels.txt"
    all_labels_df = pd.read_csv(labels_file_path)

    X_list, y_list = [], []
    files = [f for f in train_dir.iterdir() if f.is_file() and f.suffix.lower() == ".f06"]

    print(f"Extracting features from {len(files)} training images in {train_dir.name}...")

    with torch.no_grad():
        for f in files:
            img, lbl_true = load_new_format(f, all_labels_df)
            if normalize:
                sim = Simulate16BitCamera(get_or_compute_global_max(), burn_opt=1)
                img = sim(img)

            x_tensor = img.unsqueeze(0).to(dev)

            image_features = []
            for m in models:
                pred = m(x_tensor).view(-1).cpu().numpy()
                image_features.extend(pred)

            X_list.append(image_features)
            y_list.append(lbl_true.cpu().numpy())

    return np.array(X_list), np.array(y_list)


def train_and_bundle_meta_learner(X_train, y_train, save_dir: str, base_model_paths: list):
    """
    Trains the Meta-Learner on the training data and bundles it with the base models.
    """
    save_dir = rel_to_root(save_dir)
    print("\n==================================================")
    print(f" META-LEARNER TRAINING")
    print(f" Training Images: {X_train.shape[0]}")
    print(f" Base Models (Features per image): {X_train.shape[1]}")
    print("==================================================\n")

    # Train the Meta-Learner against the true values

    # meta_model = MultiOutputRegressor(
    #     RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    # )

    # The Gradient Boosting Meta-Learner
    meta_model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=100,  # Builds 100 sequential trees per label
            learning_rate=0.1,  # How aggressively it tries to fix mistakes
            max_depth=5,  # Prevents the trees from getting too complex (fights overfitting)
            random_state=42
        )
    )

    print("Training Random Forest Meta-Learner... (Building 400 trees)")
    meta_model.fit(X_train, y_train)

    # Create the Model Artifact Bundle
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, "ensemble_meta_learner.pkl")
    joblib.dump(meta_model, meta_path)

    print("\nBundling Base Models...")

    # 1. Open a new text file called 'model_order.txt' inside the bundle
    order_txt_path = os.path.join(save_dir, "model_order.txt")
    with open(order_txt_path, "w") as f:
        for path in base_model_paths:
            file_name = os.path.basename(path)
            dest_path = os.path.join(save_dir, file_name)
            shutil.copy(path, dest_path)

            # 2. Write the exact filename into the text file
            f.write(f"{file_name}\n")

    print(f"✅ Training Complete! Artifact Bundle saved securely to: {save_dir}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    # 1. Define all your Specialist CNNs here
    ckpt_paths = [
        ## Ref Index Models
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,2-16,9(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000255.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,4-9,6(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000208.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,4-17,0(noise)45/ref_multyLabelCNN_ref_e45_lr0.001_bs64_val0.000202.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,5-17,2(noise)41/ref_multyLabelCNN_ref_e41_lr0.001_bs64_val0.000198.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,2(noise)36/ref_multyLabelCNN_ref_e36_lr0.001_bs64_val0.000254.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,2(noise)41/ref_multyLabelCNN_ref_e41_lr0.001_bs64_val0.000175.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,7(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000248.pt"),
        #rel_to_root(
           # "outputs/models/multyLabelCNN/1,2-12,9(noise)49/ref_multyLabelCNN_ref_e49_lr0.001_bs64_val0.000199.pt"),

        ## Diameter Models
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0,3-1,5(noise)41/diam_multyLabelCNN_diam_e41_lr0.001_bs64_val0.000517.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0,3-2,0(noise)48/diam_multyLabelCNN_diam_e48_lr0.001_bs64_val0.000458.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0,3-2,3(noise)44/diam_multyLabelCNN_diam_e44_lr0.001_bs64_val0.000527.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0,4-2,1(noise)43/diam_multyLabelCNN_diam_e43_lr0.001_bs64_val0.000627.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0.7-4/diam_multyLabelCNN_diam_e29_lr0.001_bs32_val0.000617.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/Diameter/0.8-2/diam_multyLabelCNN_diam_e39_lr0.001_bs32_val0.000556.pt"),

        ##Ratio Models
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,2-65,9(noise)61/ratio_multyLabelCNN_ratio_e61_lr0.001_bs64_val0.057676.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,2-76,2(noise)64/ratio_multyLabelCNN_ratio_e64_lr0.001_bs64_val0.055743.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,3-71,9(noise)45/ratio_multyLabelCNN_ratio_e45_lr0.001_bs64_val0.074354.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,4-63,7(noise)36/ratio_multyLabelCNN_ratio_e36_lr0.001_bs64_val0.071812.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,4-65,1(noise)51/ratio_multyLabelCNN_ratio_e51_lr0.001_bs64_val0.065210.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,4-68,2(noise)53/ratio_multyLabelCNN_ratio_e53_lr0.001_bs64_val0.058962.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,5-61,1(noise)44/ratio_multyLabelCNN_ratio_e44_lr0.001_bs64_val0.061423.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,5-71,6(noise)48/ratio_multyLabelCNN_ratio_e48_lr0.001_bs64_val0.065814.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/ratio_newData/2,5-89,0(noise)54/ratio_multyLabelCNN_ratio_e54_lr0.001_bs64_val0.066288.pt"),

        ##Thickness Models
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/0,8-5,5(noise)50/thick_multyLabelCNN_thick_e50_lr0.001_bs64_val0.001032.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/0,8-5,6(noise)59/thick_multyLabelCNN_thick_e59_lr0.001_bs64_val0.000641.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/1,0-5,9(noise)45/thick_multyLabelCNN_thick_e45_lr0.001_bs64_val0.001079.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/1,0-7,6(noise)55/thick_multyLabelCNN_thick_e55_lr0.001_bs64_val0.000950.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/1,2-5,5(noise)35/thick_multyLabelCNN_thick_e35_lr0.001_bs64_val0.001480.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/thickness/1,4-7,3(noise)48/thick_multyLabelCNN_thick_e48_lr0.001_bs64_val0.001106.pt"),
    ]

    print(f"Loading {len(ckpt_paths)} Base Models...")
    models = [torch.load(p, map_location="cpu", weights_only=False).to(device) for p in ckpt_paths]

    # 2. Point ONLY to your physically isolated training folder
    dir_meta_train = rel_to_root("Data/test_data/rs_meta_train")

    # The folder where the .pkl and all copied .pt files will live
    save_directory = "outputs/models/Ensemble_Bundle"

    # 3. Extract the features (Runs the CNNs ONLY on the training folder)
    X_train, y_train = extract_training_features(models, dir_meta_train, normalize=True)

    # 4. Train and Bundle (No testing happens here)
    train_and_bundle_meta_learner(
        X_train=X_train,
        y_train=y_train,
        save_dir=save_directory,
        base_model_paths=ckpt_paths
    )