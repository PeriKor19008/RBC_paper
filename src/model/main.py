from src.model.experiments.cnn import *
from src.utils.paths import rel_to_root
import os
import glob
import shutil
import torch
from src.model.experiments.tests_helper import *
from src.model.experiments.test import *



def train_single_model():
    #placeholders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_path = rel_to_root(
        "outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt")
    ae_model = torch.load(ae_path, map_location="cpu", weights_only=False).to(device).eval()

    # train_CNN(32, 1, 0.001,
    #           [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
    #           [250, 128],
    #           False, True, burn_opt=1)
    train_multy_label_CNN(
        batchSize=32,
        epochs=2,
        lr_rate=0.001,
        conv_config=[("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),
                     ("conv", 240)],
        fc_config=[250, 128],
        normalize=True,
        burn_opt=1,
        target_idx=3  # ref index
    )






    #_, _, rd = run_autoencoder(32, 1, 0.001, [2048, 1024, 512, 128], 64, True)


    #_, _, rd = run_autoencoder(32, 25, 0.001, [1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [2048,1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [4000,2048,1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [1024, 512, 128, 64], 32)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [2048,1024, 512, 128,64], 32)


def run_all_four_models():
    label_names = ["Diameter", "Thickness", "Ratio", "Refractive Index"]

    for i in range(25):
        print(f"\n==================================================")
        print(f"  STARTING SPECIALIZED TRAINING FOR: Ref index, {25 + i} epochs")
        print(f"==================================================\n")

        train_multy_label_CNN(
            batchSize=64,
            epochs=20 + i,
            lr_rate=0.001,
            conv_config=[("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),
                         ("conv", 240)],
            fc_config=[250, 128],
            normalize=True,
            burn_opt=1,
            target_idx=3,
            noise=True  # ref index
        )


def run_automated(starting_epoch, idx, noise=False,):


    TARGET_IDX = idx

    # Where you want the final, nicely-named folders to live
    base_target_dir = rel_to_root("outputs/models/multyLabelCNN/RefIndex")
    os.makedirs(base_target_dir, exist_ok=True)

    for i in range(25):
        print(f"\n==================================================")
        print(f"  STARTING AUTOMATED PIPELINE: Run {i + 1}/25")
        print(f"==================================================\n")

        # ----------------------------------------------------
        # 1. TRAIN THE MODEL
        # ----------------------------------------------------
        train_losses, val_losses, run_dir = train_multy_label_CNN(
            batchSize=64,
            epochs=starting_epoch + i,
            lr_rate=0.001,
            conv_config=[("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),
                         ("conv", 240)],
            fc_config=[250, 128],
            normalize=True,
            burn_opt=1,
            target_idx=TARGET_IDX,
            noise=noise
        )

        # ----------------------------------------------------
        # 2. FIND THE SAVED MODEL
        # ----------------------------------------------------
        # Look inside the new timestamp folder for the .pt file
        pt_files = glob.glob(os.path.join(run_dir, "*.pt"))
        if not pt_files:
            print(f"[!] Could not find a .pt file in {run_dir}! Skipping testing.")
            continue

        model_path = pt_files[0]
        print(f"Found new model: {model_path}")

        # ----------------------------------------------------
        # 3. TEST THE MODEL
        # ----------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location="cpu", weights_only=False).to(device).eval()

        data_dir_good = rel_to_root("Data/test_data/rs")
        out_pct = os.path.join(run_dir, "test_graphs/single_ensemble_avg_pct_error.png")

        # Pass the single model as a list so the ensemble function accepts it
        avg_err, max_err = test_single_label_ensemble(
            models = [model],
            dir_path=data_dir_good,
            save_path_pct=out_pct,
            thresh=99,
            block=False,
            jitter=False,
            noise=False,  # Ensure noise is OFF for the test!
            normalize=True,
            trimed=False,
            target_idx=TARGET_IDX
        )

        # ----------------------------------------------------
        # 4. RENAME THE FOLDER IN PLACE
        # ----------------------------------------------------
        # Format numbers to 1 decimal place and replace '.' with ','
        avg_str = f"{avg_err:.1f}".replace(".", ",")
        max_str = f"{max_err:.1f}".replace(".", ",")
        noise_tag = "(noise)" if noise else "(no_noise)"
        new_folder_name = f"{avg_str}-{max_str}{noise_tag}{starting_epoch + i}"

        # Find the parent folder where the run_dir was originally created
        parent_dir = os.path.dirname(run_dir)
        new_dir_path = os.path.join(parent_dir, new_folder_name)

        # Anti-Collision: If a folder with this exact name already exists, add a suffix
        counter = 1
        final_dir_path = new_dir_path
        while os.path.exists(final_dir_path):
            final_dir_path = f"{new_dir_path}_{counter}"
            counter += 1

        # Rename the directory without moving it
        os.rename(run_dir, final_dir_path)
        print(f"\n✅ PIPELINE COMPLETE: Folder renamed to {final_dir_path}")



if __name__ == "__main__":
    # 0 = diameter, 1 = thickness, 2 = ratio, 3 = ref_index
    for i in range (5):
        run_automated(35,2,True)
    for i in range(5):
        run_automated(30,2, False)



    #run_all_four_models()
    # print("------------------------------------------------------------------------------------------------------------")
    # print(
    #     "------------------------------------------------------------------------------------------------------------")
    # print(
    #     "------------------------------------------------------------------------------------------------------------")
    #run_all_four_models()
    #train_single_model()

    #run_finetuning_batch()

    #multi_train_CNN()
    #train_ae_regressor_head(25,0.001,64,(128,64),"outputs/models/FCAutoencoder/20251117-162720_FCAutoencoder_e35_lr0.0008_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt")

