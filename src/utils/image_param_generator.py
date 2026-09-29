import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math

from src.utils.paths import rel_to_root


def generate_synthetic_labels(num_samples: int = 3000):
    """
    Generates random physical labels within specified continuous ranges and saves to CSV.
    """
    print(f"Generating {num_samples} sets of labels...")

    # Generate arrays of random floats uniformly distributed across your exact ranges
    diameter = np.random.uniform(low=4.500, high=10.500, size=num_samples)
    thickness = np.random.uniform(low=1.500, high=3.000, size=num_samples)
    ratio = np.random.uniform(low=0.40, high=0.80, size=num_samples)
    ref_index = np.random.uniform(low=1.010, high=1.100, size=num_samples)

    # Combine into a pandas DataFrame
    df = pd.DataFrame({
        'diameter': diameter,
        'thickness': thickness,
        'ratio': ratio,
        'ref_index': ref_index
    })

    # Optional: Round the values to 3 decimal places so they match your formatting style
    df = df.round(3)
    return df


def plot_label_distributions(df):
    print("Generating KDE density plots...")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle('Density of Synthetic Physical Parameters', fontsize=18, fontweight='bold', y=0.98)

    columns = ['diameter', 'thickness', 'ratio', 'ref_index']
    ax_list = axes.flatten()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, col in enumerate(columns):
        ax = ax_list[i]

        sns.kdeplot(
            data=df,
            x=col,
            ax=ax,
            fill=True,
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            bw_adjust=0.5
        )

        # ---------------------------------------------------------
        # NEW: Dynamically grab the min and max for the current label
        # ---------------------------------------------------------
        c_min = df[col].min()
        c_max = df[col].max()

        ax.set_title(f'{col.capitalize()} Distribution', fontsize=14)

        # Inject the min and max directly into the x-axis label
        ax.set_xlabel(f'Value  ({c_min:.3f} - {c_max:.3f})', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout()
    plt.show()


def export_labels_to_txt(df: pd.DataFrame, output_filename: str = "simulation_params.txt"):
    """
    Exports the label dataframe to a specific custom text format required by the simulation program.
    """
    print(f"Exporting {len(df)} labels to '{output_filename}'...")

    # Calculate how many digits we need for zero-padding (e.g., 3000 needs 4 digits)
    pad_len = len(str(len(df)))

    with open(output_filename, 'w') as file:
        for index, row in df.iterrows():
            # 1. Create a 1-based index (e.g., 0001, 0002)
            idx_str = str(index + 1).zfill(pad_len)

            # 2. Calculate permit (ref_index squared)
            permit = row['ref_index'] ** 2

            # 3. Extract the rest of the values
            d = row['diameter']
            tmax = row['thickness']
            thick_ratio = row['ratio']

            # 4. Construct the exact formatted string with proper decimal precision
            # permit: 6 decimals | others: 3 decimals
            line = (
                f"permit({idx_str})={permit:.6f}  ;  "
                f"d({idx_str})={d:.3f} ; "
                f"tmax({idx_str})={tmax:.3f}   ; "
                f"thick_ratio({idx_str})={thick_ratio:.3f}\n"
            )

            # 5. Write to file
            file.write(line)

    print("Export complete!")


def convert_to_clean_txt(input_filename: str = "simulation_params.txt", output_filename: str = "clean_labels.txt"):
    input_filename = rel_to_root(input_filename)
    output_filename = rel_to_root(output_filename)
    print(f"Reading from {input_filename}...")

    # This pattern looks for the numbers immediately following the "=" signs
    pattern = r"permit\(\d+\)=([\d.]+)\s*;\s*d\(\d+\)=([\d.]+)\s*;\s*tmax\(\d+\)=([\d.]+)\s*;\s*thick_ratio\(\d+\)=([\d.]+)"

    count = 0
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:

        # 1. CLEAN HEADER: Just words separated by commas. No tabs, no trailing commas.
        outfile.write("d,tmax,thick_ratio,ref_index\n")

        for line in infile:
            match = re.search(pattern, line)
            if match:
                # Extract the 4 numbers as text
                permit_str, d_str, tmax_str, ratio_str = match.groups()

                # Convert them to floats for math
                permit = float(permit_str)
                d = float(d_str)
                tmax = float(tmax_str)
                thick_ratio = float(ratio_str)

                # Calculate the refractive index
                ref_index = math.sqrt(permit)

                # 2. CLEAN DATA: Just variables separated by commas. No spaces or tabs.
                outfile.write(f"{d:.3f},{tmax:.3f},{thick_ratio:.3f},{ref_index:.3f}\n")
                count += 1

    print(f"Successfully converted {count} lines and saved to '{output_filename}'.")
if __name__ == "__main__":
    # You can change the number of samples or output file name here if needed
    # labels = generate_synthetic_labels(num_samples=3500)
    # plot_label_distributions(labels)
    # export_labels_to_txt(labels, "../../Data/test_data/test_params.txt")
    convert_to_clean_txt("Data/test_data/test_data_params.txt", "Data/test_data/clean_labels.txt")