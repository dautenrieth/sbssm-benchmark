"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Convenience script that iterates over every JSON space file in its own
    directory and renders an interactive 3-D surface plot for each one.

    - key role: Quick visual inspection tool for the 2-D test-case spaces
                (with noise) stored alongside this script.
    - dependency: space.Space, functionloader.load_functions_from_json,
                  pathlib
    - output: Interactive matplotlib 3-D surface plots (one per JSON file).

Usage:
    python vis.py
    # Visualises all *.json files found in the same directory.
"""

from pathlib import Path
from space import Space
from functionloader import load_functions_from_json

def visualize_all_spaces_in_current_folder():
    """
    Render a 3-D surface plot for every JSON space file in this directory.

    Loads each ``*.json`` file as a :class:`Space` with a 101-point grid per
    axis and calls ``plot_3d_two_varied`` with the criticality plane enabled.
    All dimensions are varied (no fixed values).
    """
    # Directory of the current script
    current_dir = Path(__file__).parent

    # All JSON files in the current directory
    json_files = list(current_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found in the current directory.")
        return

    # Problem dimension and sampling resolution (assumed: 2D with 101 points per axis)
    n_dim = 2
    n_points = 101

    for file in json_files:
        print(f"\nVisualizing: {file.name}")

        # Load function definitions from JSON file
        function_instances = load_functions_from_json(file)

        # Create search space with the specified functions
        space = Space(
            dimensions=n_dim,
            functions=function_instances,
            n_points=n_points
        )

        # 2D/3D visualization
        space.visualizer.plot_3d_two_varied(
            show_critplane=True,
            fixed_values=[None] * n_dim  # None means all variables are varied (no fixing)
        )

        print("Done:", file.name)

if __name__ == "__main__":
    visualize_all_spaces_in_current_folder()
