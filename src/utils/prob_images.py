import torch
import sys
import contextlib
import io
import re
from test import *
from src.model.experiments.tests_helper import *
from src.utils.paths import rel_to_root


# Assuming this helper exists in your setup
# If not, ensure it's imported or defined
def batch_test_outliers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of all model paths
    model_paths = [
        "outputs/models/multyLabelCNN/ratio_newData/2,2-65,9(noise)61/ratio_multyLabelCNN_ratio_e61_lr0.001_bs64_val0.057676.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,2-76,2(noise)64/ratio_multyLabelCNN_ratio_e64_lr0.001_bs64_val0.055743.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,3-71,9(noise)45/ratio_multyLabelCNN_ratio_e45_lr0.001_bs64_val0.074354.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,4-63,7(noise)36/ratio_multyLabelCNN_ratio_e36_lr0.001_bs64_val0.071812.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,4-65,1(noise)51/ratio_multyLabelCNN_ratio_e51_lr0.001_bs64_val0.065210.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,4-68,2(noise)53/ratio_multyLabelCNN_ratio_e53_lr0.001_bs64_val0.058962.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,5-61,1(noise)44/ratio_multyLabelCNN_ratio_e44_lr0.001_bs64_val0.061423.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,5-71,6(noise)48/ratio_multyLabelCNN_ratio_e48_lr0.001_bs64_val0.065814.pt",
        "outputs/models/multyLabelCNN/ratio_newData/2,5-89,0(noise)54/ratio_multyLabelCNN_ratio_e54_lr0.001_bs64_val0.066288.pt"
    ]

    all_outliers = set()
    data_dir_good = rel_to_root("Data/test_data/rs")

    print(f"Starting batch test on {len(model_paths)} models...")

    for path in model_paths:
        full_path = rel_to_root(path)
        print(f"Testing model: {path}")

        # Load the model
        model = torch.load(full_path, map_location="cpu", weights_only=False).to(device).eval()

        # Capture stdout to catch the "[!] Outlier Detected" prints
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            test_single_label(
                model, data_dir_good, "temp_graph.png", 20,
                block=False, jitter=False, noise=False, normalize=True, target_idx=2
            )

        # Parse the output for filenames
        output = f.getvalue()
        found_files = re.findall(r"\[!\] Outlier Detected: ([\w\.]+)", output)
        all_outliers.update(found_files)
        print(f"Found {len(found_files)} outliers in this model.")

    # Save to file
    with open("all_problematic_images.txt", "w") as f:
        for image in sorted(list(all_outliers)):
            f.write(f"{image}\n")

    print(f"\nDone! Found {len(all_outliers)} unique problematic images.")
    print("List saved to: all_problematic_images.txt")


if __name__ == "__main__":
    batch_test_outliers()