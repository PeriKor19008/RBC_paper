import torch.nn as nn
import torch
from src.model.training.train_logs import *
from src.model.plot import *
from src.model.training.run_dirs import *
import copy
from src.model.training.schedulers import build_scheduler, step_scheduler, current_lr


def fine_tune_single_label(model_path: str, dataloaders: dict, criterion,
                           num_epochs: int = 10, batch_size: int = 32,
                           learning_rate: float = 1e-5, target_idx: int = 3,
                           ae: nn.Module | None = None):
    print(f"\n==================================================")
    print(f"  STARTING FINE-TUNING FOR: {Path(model_path).name}")
    print(f"==================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the pre-trained winning model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.train()

    if ae is not None:
        ae.to(device)
        ae.eval()

    # 2. Setup a fresh AdamW optimizer with a TINY, flat learning rate
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-3},  # Slightly higher weight decay to prevent memorization
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )

    epoch_losses, val_losses = [], []
    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    y_mean, y_std = _compute_label_stats(dataloaders['train'], device)
    y_mean = y_mean.detach()
    y_std = y_std.detach()

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running = 0.0
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            if ae is not None:
                with torch.no_grad():
                    x = ae(x)

            optimizer.zero_grad()
            out = model(x)

            # Slice for single label
            y_single = y[:, target_idx].unsqueeze(1)
            y_mean_single = y_mean[target_idx]
            y_std_single = y_std[target_idx]

            y_norm = (y_single - y_mean_single) / y_std_single
            out_norm = (out - y_mean_single) / y_std_single
            loss = criterion(out_norm, y_norm)

            loss.backward()
            optimizer.step()
            # Notice: NO SCHEDULER STEP HERE! The LR stays perfectly flat.

            running += loss.item()

        epoch_loss = running / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)

        # --- VALIDATION ---
        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, y in dataloaders.get('val', []):
                x, y = x.to(device), y.to(device)
                if ae is not None:
                    x = ae(x)
                out = model(x)

                y_single = y[:, target_idx].unsqueeze(1)
                y_mean_single = y_mean[target_idx]
                y_std_single = y_std[target_idx]

                y_norm = (y_single - y_mean_single) / y_std_single
                out_norm = (out - y_mean_single) / y_std_single
                v += criterion(out_norm, y_norm).item()

        v /= max(1, len(dataloaders.get('val', [])))
        val_losses.append(v)

        # Save best validation weights
        if v < best_val:
            best_val = v
            best_wts = copy.deepcopy(model.state_dict())
            print(f"[{epoch + 1}/{num_epochs}] train={epoch_loss:.6f}  val={v:.6f} <-- NEW BEST", flush=True)
        else:
            print(f"[{epoch + 1}/{num_epochs}] train={epoch_loss:.6f}  val={v:.6f}", flush=True)

    # 3. Load the best weights found during fine-tuning
    model.load_state_dict(best_wts)

    # 4. Save the new model in the same directory, but append "_finetuned"
    original_dir = os.path.dirname(model_path)
    original_name = os.path.basename(model_path).replace(".pt", "")
    new_save_path = os.path.join(original_dir, f"{original_name}_finetuned_val{best_val:.6f}.pt")

    model_cpu = copy.deepcopy(model).to("cpu")
    torch.save(model_cpu, new_save_path)

    print(f"Saved Fine-Tuned Model to: {new_save_path}")
    return new_save_path

def train_model_single_label(model, dataloaders, criterion, optimizer,
                         num_epochs, batch_size, learning_rate, layers=None,
                         conv_config=None, fc_config=None,
                         scheduler_name: str | None = None, scheduler_params: dict | None = None,
                         selection="val_loss", ae: nn.Module | None = None,
                         target_idx: int = 0,seed=None):

    print(selection)

    ### MODIFIED: Added label names for clean saving
    label_names = ["diam", "thick", "ratio", "ref"]
    target_name = label_names[target_idx]
    print(f"--- Training Specialized CNN for: {target_name.upper()} ---")

    steps_per_epoch = len(dataloaders['train'])
    scheduler, sched_mode = build_scheduler(
        optimizer,
        scheduler_name,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=learning_rate,
        **(scheduler_params or {})
    )

    if scheduler_name:
        mx = (scheduler_params or {}).get("max_lr", None)
        lr_tag = f"{scheduler_name}, max={mx:.2e}" if mx is not None else str(scheduler_name)
    else:
        lr_tag = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ae is not None:
        ae.to(device)
        ae.eval()

    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers,
        extra_info={"conv_config": conv_config, "fc_config": fc_config},seed=seed
    )

    print("using device:", device, "and scheduler:", scheduler_name or "none")
    epoch_losses, val_losses = [], []
    epoch_lr = []
    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    # This still calculates the mean/std for all 4 labels
    y_mean, y_std = _compute_label_stats(dataloaders['train'], device)
    y_mean = y_mean.detach()
    y_std = y_std.detach()

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            if ae is not None:
                with torch.no_grad():
                    x = ae(x)
            optimizer.zero_grad()
            out = model(x)  # out shape is now [Batch_Size, 1]

            ### MODIFIED: Slice the target and stats to match the 1 output
            y_single = y[:, target_idx].unsqueeze(1)
            y_mean_single = y_mean[target_idx]
            y_std_single = y_std[target_idx]

            # normalized loss (keep optimization invariant to label scales)
            y_norm = (y_single - y_mean_single) / y_std_single
            out_norm = (out - y_mean_single) / y_std_single
            loss = criterion(out_norm, y_norm)
            loss.backward()
            optimizer.step()

            # per-step schedulers
            step_scheduler(scheduler, sched_mode)

            running += loss.item()

        epoch_loss = running / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)

        # val
        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, y in dataloaders.get('val', []):
                x, y = x.to(device), y.to(device)
                if ae is not None:
                    x = ae(x)
                out = model(x)

                ### MODIFIED: Slice the validation target as well
                y_single = y[:, target_idx].unsqueeze(1)
                y_mean_single = y_mean[target_idx]
                y_std_single = y_std[target_idx]

                y_norm = (y_single - y_mean_single) / y_std_single
                out_norm = (out - y_mean_single) / y_std_single
                v += criterion(out_norm, y_norm).item()
        v /= max(1, len(dataloaders.get('val', [])))
        val_losses.append(v)

        # per-epoch schedulers
        if sched_mode in {"epoch", "plateau"}:
            step_scheduler(scheduler, sched_mode, val_loss=v)

        current_lr_val = optimizer.param_groups[0]["lr"]
        epoch_lr.append(current_lr_val)

        # selection: lowest val loss
        if v < best_val:
            best_val = v
            best_wts = copy.deepcopy(model.state_dict())
            print(f"[{epoch + 1}/{num_epochs}] train={epoch_loss:.6f}  val={v:.6f}  lr={current_lr_val:.2e}",
                  flush=True)

    model.load_state_dict(best_wts)

    ### MODIFIED: Save files with the specific label name so they don't overwrite
    if selection == "val_loss":
        ckpt = os.path.join(
            run_dir,
            f"{label_names[target_idx]}_{arch_name}_{target_name}_e{num_epochs}_lr{learning_rate}_bs{batch_size}_val{min(val_losses):.6f}.pt"
        )
    else:
        ckpt = os.path.join(
            run_dir,
            f"{label_names[target_idx]}_{arch_name}_{target_name}_e{num_epochs}_lr{learning_rate}_bs{batch_size}_pct{best_val:.3f}.pt"
        )

    model_cpu = copy.deepcopy(model).to("cpu")
    torch.save(model_cpu, ckpt)

    # per-run combined plot
    plot_loss_graphs(epoch_losses, val_losses, run_number=1, num_epochs=num_epochs,
                     learning_rate=learning_rate, batch_size=batch_size,
                     layers=layers, out_dir=figs_dir, lr_tag=lr_tag)

    # per-run log
    run_log_path = os.path.join(run_dir, f"run_log_{target_name}.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss=best_val, device=device,
                    epoch_losses=epoch_losses, val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir,
                    scheduler_name=scheduler_name, scheduler_params=scheduler_params, epoch_lrs=epoch_lr)

    return epoch_losses, val_losses, run_dir







def _compute_label_stats(train_loader, device):
    s = torch.zeros(4, device=device)
    ss = torch.zeros(4, device=device)
    n = 0
    with torch.no_grad():
        for _, y in train_loader:
            y = y.to(device)
            s  += y.sum(dim=0)
            ss += (y ** 2).sum(dim=0)
            n  += y.size(0)
    mean = s / n
    var  = (ss / n) - mean**2
    std  = var.clamp_min(1e-8).sqrt()
    return mean, std






