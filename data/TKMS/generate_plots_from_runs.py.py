"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Aggregates TKMS simulation results from per-scenario JSON reports into a
    single DataFrame and renders 3-D surface plots (PNG/PDF/EPS) for every
    metric–parameter–parameter combination.

    - key role: Post-processing entry point for the TKMS maritime simulation
                dataset; produces publication-ready 3-D surface visualisations.
    - dependency: os, json, pandas, numpy, matplotlib, tqdm
    - output: ``dataframettc.csv`` (aggregated data cache) and
              ``plots_combined_surface/`` directory with .png/.pdf/.eps files.

Usage:
    # Place JSON simulation reports in the same directory, then run:
    python generate_plots_from_runs.py.py

    # Edit `selected_parameters` and `labels` to match your scenario variables.
    # Edit `metrics` to select which output metrics to visualise.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

import matplotlib as mpl

# === STYLE / EXPORT SETTINGS ===
mpl.rcParams.update({
    "pdf.fonttype": 42,   # TrueType embedding
    "ps.fonttype": 42,    # TrueType embedding for PS/EPS
    # Optional:
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
})

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
})

# === SETUP ===
folder_path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(folder_path, "dataframettc.csv")

# === DATA LOADING OR CREATION ===
if os.path.exists(output_file):
    df = pd.read_csv(output_file)
else:
    data = []
    json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]

    for file_name in tqdm(json_files, desc="Processing JSON files"):
        with open(os.path.join(folder_path, file_name), "r") as file:
            content = json.load(file)

            summary_metrics = {
                m["name"]: m["value"]
                for m in content.get("summary", {}).get("max_metrics", [])
            }

            min_ttc = summary_metrics.get("ttcMetric", None)

            cri_metric_dict = summary_metrics.get("criMetric", None)
            dcpa = tcpa = None
            if isinstance(cri_metric_dict, dict):
                dcpa = cri_metric_dict.get("dcpa", None)
                tcpa = cri_metric_dict.get("tcpa", None)

            cri_metric = summary_metrics.get("criMetric", None) if not isinstance(cri_metric_dict, dict) else None

            ship_domain_overlap = summary_metrics.get(
                "ShipDomainOverlap", summary_metrics.get("shipDomainOverlap", None)
            )

            # Extract scenario parameters from filename
            file_parameters = file_name.replace("report_scenario_", "").split("#")
            parameters = {p.split("@")[0]: p.split("@")[1] for p in file_parameters if "@" in p}

            parameters.update({
                "min_ttc": min_ttc,
                "cri_metric": cri_metric,
                "dcpa": dcpa,
                "tcpa": tcpa,
                "ship_domain_overlap": ship_domain_overlap,
            })

            data.append(parameters)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# === PLOTTING FUNCTION ===
def plot_3d_surface(
    df, x_col, y_col, z_col,
    x_label, y_label, z_label,
    filename,
    show_title=False,
    show_colorbar=False,
    label_colorbar=False
):
    """
    Render a 3-D surface plot from a pivoted DataFrame and save it as PNG/PDF/EPS.

    The DataFrame is pivoted so that ``x_col`` forms the columns, ``y_col`` the
    rows, and ``z_col`` the surface height. Each (x, y) combination must be
    unique; duplicate entries will cause the pivot to fail.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the three specified columns.
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
    try:
        pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
    except ValueError as e:
        print(f"Pivoting failed: {e}")
        print(f"Make sure each combination of {x_col} and {y_col} is unique (no duplicates).")
        return

    if pivot_df.empty:
        print(f"No data for plot {filename}, skipping...")
        return

    X_vals = pivot_df.columns.values
    Y_vals = pivot_df.index.values
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivot_df.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # alpha=1.0 avoids EPS transparency artefacts
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=1.0)

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_zlabel(z_label, labelpad=12)
    ax.zaxis.set_tick_params(pad=5)

    if show_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        if label_colorbar:
            cbar.set_label(z_label)

    if show_title:
        ax.set_title(f"{z_label} as a function of {x_label} and {y_label}")

    # tight_layout is unreliable for 3D axes; bbox_inches='tight' is more robust
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)

    base = os.path.splitext(filename)[0]
    plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.5)
    plt.savefig(base + ".eps", format="eps", bbox_inches="tight", pad_inches=0.5)

    plt.close()
    print(f"Saved plot: {base} (.png/.pdf/.eps)")

# === METRICS / AXES ===
metrics = [
    "min_ttc",
    "max_ttc_metric",
    "cri_metric",
    "dcpa",
    "tcpa",
    "ship_domain_overlap",
]

selected_parameters = ["scenario_boats_1_geopos_latitude", "boats_1_safetyradius"]

# Labels (adjust to match your scenario variables; currently illustrative)
labels = {
    "scenario_boats_1_geopos_latitude": r"Initial Latitude Obstacle, $\varphi$ (deg)",
    "boats_1_safetyradius": r"Size Obstacle, $r_s$ (m)",

    "min_ttc": r"Min. time-to-collision, TTC (s)",
    "max_ttc_metric": r"Max. TTC metric (–)",
    "cri_metric": r"CRI metric (–)",
    "dcpa": r"DCPA (m)",
    "tcpa": r"TCPA (s)",
    "ship_domain_overlap": r"Ship-domain overlap, $R$ (–)",
}

if not all(p in df.columns for p in selected_parameters):
    raise ValueError("One or both selected parameters are missing in the DataFrame.")

# Convert axes to float
df[selected_parameters[0]] = df[selected_parameters[0]].astype(float)
df[selected_parameters[1]] = df[selected_parameters[1]].astype(float)

# Coerce metrics to numeric — more robust against CSV/JSON type artefacts
for m in metrics:
    if m in df.columns:
        df[m] = pd.to_numeric(df[m], errors="coerce")

other_params = [col for col in df.columns if col not in selected_parameters + metrics]
fixable_params = [col for col in other_params if df[col].nunique() > 1]
combinations = list(product(*[df[col].unique() for col in fixable_params]))

plot_dir = os.path.join(folder_path, "plots_combined_surface")
os.makedirs(plot_dir, exist_ok=True)

x = selected_parameters[0]
y = selected_parameters[1]

for combo in combinations:
    filter_dict = dict(zip(fixable_params, combo))
    df_filtered = df

    for k, v in filter_dict.items():
        df_filtered = df_filtered[df_filtered[k] == v]

    if df_filtered.empty:
        continue

    for metric in metrics:
        if metric not in df_filtered.columns:
            continue

        # Skip if the entire Z column is NaN (avoids empty surfaces)
        if df_filtered[metric].notna().sum() == 0:
            continue

        suffix = "__".join([f"{k}_{v}" for k, v in filter_dict.items()])
        filename = f"{metric}_{x}_{y}__{suffix}.png"
        filepath = os.path.join(plot_dir, filename)

        plot_3d_surface(
            df_filtered,
            x, y, metric,
            labels.get(x, x),
            labels.get(y, y),
            labels.get(metric, metric),
            filepath,
            show_title=False,          # off by default
            show_colorbar=False,       # optional
            label_colorbar=False
        )

print("All combinations have been successfully plotted as 3D surfaces.")