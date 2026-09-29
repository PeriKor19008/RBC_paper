import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.colors import yellow

from src.utils.paths import rel_to_root

def plot_a():
    # The EXACT 10 problematic images identified
    bad_d = [4.608, 5.399, 4.550, 4.519, 4.588, 4.764, 4.786, 4.728, 4.514, 4.652]
    bad_ratio = [0.409, 0.434, 0.419, 0.414, 0.468, 0.418, 0.423, 0.785, 0.404, 0.489]

    try:
        # Load the full background dataset
        df = pd.read_csv(rel_to_root('Data/test_data/clean_labels.txt'))

        plt.figure(figsize=(10, 6))

        # Plot the full background in faded blue
        plt.scatter(df['d'], df['thick_ratio'], alpha=0.3, color='#4C72B0', edgecolor='white', linewidth=0.2,
                    label='Total Dataset (~1500 images)')

        # Plot ONLY the 10 specific bad points in bright red
        plt.scatter(bad_d, bad_ratio, color='red', edgecolor='black', linewidth=1.5, s=80,
                    label='The 10 Failed Images (>20% Error)', zorder=5)

        # Add the threshold lines we discussed
        plt.axvline(x=4.8, color='darkred', linestyle='--', linewidth=2, label='Proposed Diameter Limit (4.8 µm)')
        plt.axhline(y=0.45, color='darkorange', linestyle='--', linewidth=2, label='Proposed Ratio Limit (0.45)')

        plt.title('Corrected Distribution: Exactly 10 Failed Images vs. Full Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Diameter ($\mu$m)', fontsize=12)
        plt.ylabel('Thickness Ratio ($n = t_{min}/t_{max}$)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        plt.savefig('corrected_diameter_vs_ratio.png', dpi=200)
        print("Saved corrected_diameter_vs_ratio.png")
    except Exception as e:
        print(f"Error: {e}")


import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import rel_to_root


def get_problematic_data(df, red_files, orange_files):
    """Helper to extract dataframes based on filenames."""

    def get_indices(file_list):
        return [int(f.split('1a.f06')[0]) - 1 for f in file_list]

    return df.iloc[get_indices(red_files)], df.iloc[get_indices(orange_files)]


def plot_diameter_vs_ratio(df, red_data, orange_data):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['d'], df['thick_ratio'], alpha=0.3, color='#4C72B0', edgecolor='none', label='Healthy')

    plt.scatter(red_data['d'], red_data['thick_ratio'], color='red', edgecolor='black', linewidth=1, s=60,
                label='Problematic (Error > 20%)', zorder=6)
    plt.scatter(orange_data['d'], orange_data['thick_ratio'], color='orange', edgecolor='black', linewidth=1, s=60,
                label='Problematic (Error > 8%)', zorder=5)


    # Adding the lines (scaled to micrometers)
    plt.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Threshold 5.0 µm')
    plt.axvline(x=5.8, color='orange', linestyle='--', linewidth=2, label='Threshold 5.5 µm')

    plt.title('Diameter vs Thickness Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Diameter ($d$ in µm)', fontsize=12)
    plt.ylabel('Ratio ($thick\_ratio$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('diameter_vs_ratio_error.png', dpi=200)
    print("Graph: Diameter vs Ratio saved with threshold lines.")


def plot_diameter_vs_ref_index(df, red_data, orange_data):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['d'], df['ref_index'], alpha=0.3, color='#4C72B0', edgecolor='none', label='Healthy')

    plt.scatter(red_data['d'], red_data['ref_index'], color='red', edgecolor='black', linewidth=1, s=60,
                label='Problematic (Error > 20%)', zorder=6)
    plt.scatter(orange_data['d'], orange_data['ref_index'], color='orange', edgecolor='black', linewidth=1, s=60,
                label='Problematic (Error > 8%)', zorder=5)


    # Adding the lines (scaled to micrometers)
    plt.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Threshold 5.000 µm')
    plt.axvline(x=5.8, color='orange', linestyle='--', linewidth=2, label='Threshold 5.500 µm')
    plt.axvline(x=10, color='yellow', linestyle='--', linewidth=2, label='Threshold 10.000 µm')

    plt.axhline(y=1.015, color='green', linestyle='--', linewidth=2, label='Threshold 1.015 Ref. Index')


    plt.title('Diameter vs Refractive Index', fontsize=14, fontweight='bold')
    plt.xlabel('Diameter ($d$ in µm)', fontsize=12)
    plt.ylabel('Refractive Index', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('diameter_vs_refindex_error.png', dpi=200)
    print("Graph: Diameter vs Refractive Index saved with threshold lines.")

# --- Main Controller ---
def generate_all_plots():


    red_files = [

        # Add your new filenames here:

        "18291a.f06", "22841a.f06", "21631a.f06", "22261a.f06",

        "10581a.f06", "16281a.f06", "08931a.f06", "05761a.f06",

        "34511a.f06", "22711a.f06"

    ]
    orange_files = [

        "30931a.f06", "12991a.f06", "14481a.f06", "21581a.f06",

        "14981a.f06", "05771a.f06", "27731a.f06", "11181a.f06",

        "34231a.f06", "26901a.f06", "31211a.f06", "01201a.f06"

    ]



    try:
        df = pd.read_csv(rel_to_root('Data/test_data/clean_labels.txt'))
        red_data, orange_data = get_problematic_data(df, red_files, orange_files)

        plot_diameter_vs_ratio(df, red_data, orange_data)
        plot_diameter_vs_ref_index(df, red_data, orange_data)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_all_plots()