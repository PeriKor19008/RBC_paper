import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os

class Simulate8BitCamera(nn.Module):
    """
    Simulates an 8-bit camera by performing per-image min-max scaling,
    quantizing to 256 discrete levels (0-255), and scaling back to [0.0, 1.0].
    """
    def __init__(self,burn : bool = False):
        super().__init__()
        self.burn = burn
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pix = [0,1,2,3,4,5,50,51,52,53,54,55,100,101,102,103,104,105,150,151,152,153,154,155,200,201,202,203,204,205]
        if self.burn:
            for i in pix:
                img[i]=0
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32)


        img = img.float()
        # 1. Find the min and max values of the specific image
        img_min = img.min()
        img_max = img.max()

        # Prevent division by zero if the image is completely flat
        if img_max - img_min == 0:
            return torch.zeros_like(img)

        # 2. Min-Max Scaling to [0.0, 1.0]
        img_normalized = (img - img_min) / (img_max - img_min)

        # 3. Scale to 0-255 and round to nearest integer (Quantization)
        # This step forces the continuous values into 256 discrete bins
        img_8bit = torch.round(img_normalized * 255.0)

        # 4. Normalize back to [0.0, 1.0] for stable CNN training
        img_cnn_ready = img_8bit / 255.0

        return img_cnn_ready


class Simulate16BitCamera(nn.Module):
    """
    Simulates a 16-bit camera.
    1) Takes the global_max (from all images) as an argument.
    2) Maps the global max to 2/3 of the 16-bit range.
    3) Returns a tensor containing only integer values.
    """

    def __init__(self, global_max: float, global_min: float = 0.0, burn: bool = False,burn_opt: int = 0):
        super().__init__()
        self.burn = burn
        self.burn_opt = burn_opt
        # 1) Take the max from all images as an arg
        self.global_max = float(global_max)
        self.global_min = float(global_min)

        # 16-bit encoding ranges from 0 to 65535
        self.max_16bit = 65535.0

        # Map max to 2/3 of the 16-bit range (43690)
        self.target_scale = self.max_16bit * (2.0 / 3.0)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        else:
            img = img.float()  # Keep float for the math steps

        pix = get_burned_pixels(burn_option = self.burn_opt)
        if self.burn:
            # 1. Save the original dimensions (e.g., [1, 50, 50])
            original_shape = img.shape

            # 2. Flatten the image to a pure 1D array so our indices work perfectly
            img = img.view(-1)

            # 3. Burn the pixels
            for i in pix:
                img[i] = 0

            # 4. Snap the image back to its original dimensions
            img = img.view(original_shape)

        # Normalize based on the global min/max
        range_val = self.global_max - self.global_min
        if range_val == 0:
            range_val = 1e-8

        img_normalized = (img - self.global_min) / range_val
        img_normalized = torch.clamp(img_normalized, 0.0, 1.0)

        # Scale up to 2/3 of the 16-bit range and round
        img_16bit = torch.round(img_normalized * self.target_scale)

        # 2) Convert all values of the image to strictly integers
        img_int = img_16bit.to(torch.int32)

        return img_int.float()


def get_or_compute_global_max(dataset=None, filename="dataset_meta.json", batch_size=32) -> float:
    """
    Finds or computes the global max, automatically saving/loading the JSON
    in the project's root directory.
    """
    # 1. Get the directory of THIS exact file (src/utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Step up two levels to reach the project root (RBC/)
    # current_dir -> src/utils
    # level 1 up  -> src
    # level 2 up  -> RBC (Project Root)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # 3. Combine the project root with the filename
    meta_filepath = os.path.join(project_root, filename)

    print(f"Checking for meta file at: {meta_filepath}")

    # 4. Check if the file exists at the root
    if os.path.exists(meta_filepath):
        with open(meta_filepath, 'r') as f:
            data = json.load(f)
            return float(data['global_max'])

    # 5. Safety check: If file is missing and we have no dataset, crash gracefully
    if dataset is None:
        raise FileNotFoundError(
            f"Error: '{meta_filepath}' not found! "
            "You must run a script that loads the dataset first to generate this file."
        )

    # 6. Calculate it if it doesn't exist
    print("Meta file not found. Scanning dataset for the first time...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    global_max = float('-inf')

    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_max = images.max().item()
            if batch_max > global_max:
                global_max = batch_max

    # 7. Save it to the project root for next time
    with open(meta_filepath, 'w') as f:
        json.dump({"global_max": global_max}, f)

    print(f"Global max calculated and saved to {meta_filepath}: {global_max}")
    return float(global_max)


def get_burned_pixels(burn_option: int = 0) -> list:
    """
    Returns a list of indices representing burned pixels.
    Assumes the image is a flattened 1D array with a width of 50 pixels.
    """
    img_width = 50
    burned_pixels = []

    # 1. Determine the size of the burned square based on the option
    if burn_option == 0:
        square_size = 10
    elif burn_option == 1:
        square_size = 5
    else:
        # Fallback: return empty list if an unknown option is passed
        return []

    # 2. Generate the indices using the selected square size
    for y in range(square_size):
        for x in range(square_size):
            index = (y * img_width) + x
            burned_pixels.append(index)

    return burned_pixels
