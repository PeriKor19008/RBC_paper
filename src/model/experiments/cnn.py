from torch import optim
from Data.DB_setup.db_config import DB_CONFIG
from torch.utils.data import DataLoader, random_split
from src.model.model import *
from src.model.training.loops import *
from datetime import datetime
from src.model.training.run_dirs import *
from src.model.noise import *
from src.utils.norm_cam_transform import *
from src.model.MAPE_Loss import *
from src.utils.paths import rel_to_root


def run_finetuning_batch():
    # 1. Paste the paths to your 20 successful ref_index models here
    winning_models = [
        # rel_to_root(
        #     "outputs/models/multyLabelCNN/RefIndex/1,3-24,9/ref_multyLabelCNN_ref_e34_lr0.001_bs32_val0.000585.pt"),
        # rel_to_root(
        #     "outputs/models/multyLabelCNN/RefIndex/1,3-24,9/ref_multyLabelCNN_ref_e34_lr0.001_bs32_val0.000585.pt"),
        # rel_to_root(
        #     "outputs/models/multyLabelCNN/RefIndex/1,3-24,9/ref_multyLabelCNN_ref_e34_lr0.001_bs32_val0.000585.pt"),
        rel_to_root("outputs/models/multyLabelCNN/RefIndex/1,3-24,9/ref_multyLabelCNN_ref_e34_lr0.001_bs32_val0.000585.pt"),
        # ... add all 20 paths ...
    ]

    # Setup your dataset exactly like you do in train_CNN
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    global_max = get_or_compute_global_max(full_dataset)
    t_layers = [Simulate16BitCamera(global_max=global_max, burn=True, burn_opt=1)]
    v_layers = [Simulate16BitCamera(global_max=global_max, burn=True, burn_opt=1)]

    train_dataset = WithTransform(train_ds, transform=nn.Sequential(*t_layers))
    val_dataset = WithTransform(val_ds, transform=nn.Sequential(*v_layers))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    criterion = nn.SmoothL1Loss(beta=0.1)

    # 2. Run the fine-tuning loop!
    for model_path in winning_models:
        fine_tune_single_label(
            model_path=model_path,
            dataloaders=dataloaders,
            criterion=criterion,
            num_epochs=10,  # Keep it short! 10 epochs is plenty to micro-step.
            learning_rate=3e-5,  # Microscopic learning rate
            target_idx=3  # 3 = ref_index
        )


def train_multy_label_CNN(batchSize, epochs, lr_rate, conv_config, fc_config=None, noise: bool = False,
              normalize: bool = False, ae: nn.Module | None = None, burn_opt: int = 0,
              log_image: bool = False, target_idx: int = 0):  # <--- MODIFIED: Added target_idx
    cur_seed = int(time.time())
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=log_image)

    # ---create datasets---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_dataset = train_ds
    val_dataset = val_ds
    global_max = get_or_compute_global_max(full_dataset)

    # --- Dynamically build transform layers ---
    t_layers = []  # Training transforms
    v_layers = []  # Validation transforms
    if normalize == 1:
        t_layers.append(Simulate16BitCamera(global_max=global_max, burn=True, burn_opt=burn_opt))
        #t_layers.append(ScaleDown())  # <--- Don't forget this math fix!

        v_layers.append(Simulate16BitCamera(global_max=global_max, burn=True, burn_opt=burn_opt))
        #v_layers.append(ScaleDown())  # <--- Don't forget this math fix!

    if noise:
        t_layers.append(AddGaussianNoise(pct=0.3, p=0.5))


    #  Wrap lists into nn.Sequential and apply to datasets
    if t_layers:
        train_dataset = WithTransform(train_ds, transform=nn.Sequential(*t_layers))
    if v_layers:
        val_dataset = WithTransform(val_ds, transform=nn.Sequential(*v_layers))

    # -----create dataloader----
    batch_size = batchSize
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    # defaults
    if fc_config is None:
        fc_config = [128]

    #  Save the current random state so we don't mess up the dataloaders
    original_rng_state = torch.get_rng_state()

    #  Force PyTorch to use your new seed
    torch.manual_seed(cur_seed)

    #  BUILD THE MODEL (The weights are now locked to cur_seed!)
    model = multyLabelCNN(conv_config, fc_config)

    #  Restore the random state so the dataloader shuffles naturally
    torch.set_rng_state(original_rng_state)



    # criterion = MAPELoss()
    criterion = nn.SmoothL1Loss(beta=0.1)
    learning_rate = lr_rate
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )

    label = f"conv{len(conv_config)}_fc{len(fc_config)}"

    # --- train ---
    train_losses, val_losses, run_dir = train_model_single_label(  # <--- MODIFIED: Called your renamed method
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        batch_size=batchSize,
        learning_rate=learning_rate,
        layers=label,
        conv_config=conv_config,
        fc_config=fc_config,
        scheduler_name="onecycle",
        scheduler_params={
            "max_lr": learning_rate * 5.0,  # or x10
            "pct_start": 0.3,
            "div_factor": (learning_rate * 5.0) / learning_rate,  # == 5.0
            "final_div_factor": 1e4,
            "cycle_momentum": False
        },
        ae=ae,
        target_idx=target_idx,  # <--- MODIFIED: Passed the index down to the loop
        seed=cur_seed
    )

    return train_losses, val_losses, run_dir

