"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Reads the DLR maritime simulation dataset from a CSV file and generates
    3-D surface plots (PNG/PDF/EPS) for every combination of two varied input
    parameters and each output metric, while holding the remaining inputs fixed.

    - key role: Post-processing entry point for the DLR dataset; produces
                publication-ready 3-D surface visualisations for all
                input-parameter pairs.
    - dependency: pandas, numpy, matplotlib, os, itertools
    - output: ``plots/`` directory containing .png/.pdf/.eps files named by
              metric, varied parameters, and fixed-parameter values.

Usage:
    # Place parameters_and_results.csv in the working directory, then run:
    python generate_plots_from_csv.py

    # Edit `input_params`, `output_params`, and `labels` to match your dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product, combinations

import matplotlib as mpl

mpl.rcParams.update({
    # Embed TrueType fonts (Type 42) in PDF/PS(EPS)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Optional: use a serif font closer to IEEE style (if available)
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
})

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
})

def plot_3d_surface(df, x_col, y_col, z_col, x_label, y_label, z_label, filename, show_title = False, show_colorbar=False, label_colorbar=False):
    """
    Render a 3-D surface plot from a pivoted DataFrame and save it as PNG/PDF/EPS.

    The DataFrame is pivoted so that ``x_col`` forms the columns, ``y_col`` the
    rows, and ``z_col`` the surface height. Each (x, y) combination must be
    unique; duplicate entries will cause the pivot to fail.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame containing at least the three specified columns.
    x_col : str
        Column name for the x-axis (pivot columns).
    y_col : str
        Column name for the y-axis (pivot index).
    z_col : str
        Column name for the surface height values.
    x_label : str
        Axis label for the x-axis.
    y_label : str
        Axis label for the y-axis.
    z_label : str
        Axis label for the z-axis (also used for the optional colour bar).
    filename : str
        Output file path (including ``.png`` extension). PDF and EPS copies
        are saved automatically alongside with the same base name.
    show_title : bool, optional
        If True, add an auto-generated title to the axes.
    show_colorbar : bool, optional
        If True, attach a colour bar to the figure.
    label_colorbar : bool, optional
        If True and ``show_colorbar`` is True, label the colour bar with
        ``z_label``.
    """
    # Pivot the data into a 2D grid: rows = y values, cols = x values, values = z
    try:
        pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
    except ValueError as e:
        print(f"Pivoting failed: {e}")
        print(f"Make sure each combination of {x_col} and {y_col} is unique (no duplicates).")
        return

    if pivot_df.empty:
        print(f"No data for plot {filename}, skipping...")
        return

    # Build meshgrid for surface plotting
    X_vals = pivot_df.columns.values
    Y_vals = pivot_df.index.values
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivot_df.values

    # Create plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Axis labels (custom display labels)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_zlabel(z_label, labelpad=12)
    ax.zaxis.set_tick_params(pad=5)

    # Colorbar uses the same label as Z
    if show_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        if label_colorbar:
            cbar.set_label(z_label)

    # Use display labels in the title, keep raw z_col in case you want to know what was plotted
    if show_title:
        plt.title(f"{z_label} Surface Plot: {z_col} as a function of {x_label} and {y_label}")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    base = os.path.splitext(filename)[0]  # strip .png
    plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.5)
    plt.savefig(base + ".eps", format="eps", bbox_inches="tight", pad_inches=0.5)
    plt.close()
    print(f"Saved plot: {base}")


def main():
    """
    Load the DLR dataset and generate 3-D surface plots for all parameter combinations.

    Iterates over every unordered pair of input parameters as (x, y) axes,
    fixes the remaining inputs at each of their unique values, and saves one
    surface plot per output metric.
    """
    df = pd.read_csv("parameters_and_results.csv")

    input_params = ["os_speed", "ts_speed", "os_maneuverability", "ts_maneuverability"]
    output_params = ["ttc", "metric", "fuzzy"]

    labels = {
        "os_speed": r"Own-ship speed, $v_{\mathrm{OS}}$ (m/s)",
        "ts_speed": r"Target-ship speed, $v_{\mathrm{TS}}$ (m/s)",
        "os_maneuverability": r"Own-ship maneuverability, $m_{\mathrm{OS}}$ (–)",
        "ts_maneuverability": r"Target-ship maneuverability, $m_{\mathrm{TS}}$ (–)",
        "ttc": r"Time to collision, TTC (s)",
        "metric": r"Ship-domain overlap , $R$ (–)",
        "fuzzy": r"Fuzzy estimate, $\mu$ (–)",
    }

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all unordered pairs of input parameters
    for x_param, y_param in combinations(input_params, 2):

        # The rest are fixed
        fixed_params = [p for p in input_params if p not in (x_param, y_param)]

        # Unique values for the fixed params
        unique_values = {p: sorted(df[p].unique()) for p in fixed_params}
        fixed_combos = list(product(*[unique_values[p] for p in fixed_params]))

        print(f"\nX: {x_param} | Y: {y_param} | Fixed: {fixed_params} "
              f"| #fixed-combos: {len(fixed_combos)}")

        for combo in fixed_combos:
            fixed_values = dict(zip(fixed_params, combo))

            df_filtered = df
            for p, v in fixed_values.items():
                df_filtered = df_filtered[df_filtered[p] == v]

            if df_filtered.empty:
                continue

            for z_param in output_params:
                x_label = labels.get(x_param, x_param)
                y_label = labels.get(y_param, y_param)
                z_label = labels.get(z_param, z_param)

                suffix = "_".join(f"{p}_{fixed_values[p]}" for p in fixed_params)

                filename = os.path.join(
                    output_dir,
                    f"DLR_plot_{z_param}_{x_param}_{y_param}__{suffix}.png"
                )

                plot_3d_surface(
                    df_filtered,
                    x_param, y_param, z_param,
                    x_label, y_label, z_label,
                    filename
                )


if __name__ == "__main__":
    main()