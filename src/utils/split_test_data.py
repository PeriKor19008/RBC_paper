import os
import random
import shutil
from pathlib import Path

from src.utils.paths import rel_to_root


def physically_split_test_data(source_dir: str, num_meta_images: int = 2000):
    source_dir = rel_to_root(source_dir)
    source_path = Path(source_dir).resolve()
    parent_dir = source_path.parent

    # Define new directories
    meta_train_dir = parent_dir / f"{source_path.name}_meta_train"
    final_test_dir = parent_dir / f"{source_path.name}_final_test"

    # Create the directories
    meta_train_dir.mkdir(exist_ok=True)
    final_test_dir.mkdir(exist_ok=True)

    # Get all .f06 files
    all_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() == ".f06"]

    print(f"Found {len(all_files)} total .f06 files.")

    # Randomly shuffle the files for an unbiased split
    random.seed(42)  # Ensures reproducibility
    random.shuffle(all_files)

    # Split the lists
    meta_files = all_files[:num_meta_images]
    test_files = all_files[num_meta_images:]

    # --- CHANGED: Now using shutil.copy instead of shutil.move ---
    print(f"Copying {len(meta_files)} files to {meta_train_dir.name}...")
    for f in meta_files:
        shutil.copy(str(f), str(meta_train_dir / f.name))

    print(f"Copying {len(test_files)} files to {final_test_dir.name}...")
    for f in test_files:
        shutil.copy(str(f), str(final_test_dir / f.name))

    print("\n✅ Non-destructive split complete! Your original folder is untouched.")


if __name__ == "__main__":
    # Point this to your current folder containing the 3,500 images
    target_folder = "Data/test_data/rs"
    physically_split_test_data(target_folder, 2000)