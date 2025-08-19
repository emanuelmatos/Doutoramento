#!/usr/bin/env python
# coding: utf-8

"""
Contingency Tables Visualization Script
Source: https://moonbooks.org/Articles/How-to-create-and-plot-a-contingency-table-or-crosstab-from-two-dataframe-columns-using-pandas-in-python/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configsite import filesdir  # Custom configuration for file directory


def set_working_directory(directory):
    """
    Set the working directory.

    Args:
        directory (str): Path to the working directory.

    Returns:
        None
    """
    os.chdir(directory)
    print(f"Working directory set to: {directory}")


def load_dataset(filename):
    """
    Load the dataset from a file.

    Args:
        filename (str): Path to the file.

    Returns:
        DataFrame: Loaded pandas DataFrame.
    """
    try:
        df = pd.read_csv(filename, delimiter=',')
        print("Data loaded successfully:")
        print(df.head())
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading file '{filename}': {e}")


def plot_confusion_matrix(df, col1, col2, output_file):
    """
    Generate and plot a contingency table (heatmap) between two columns.

    Args:
        df (DataFrame): Input DataFrame.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.
        output_file (str): Path to save the heatmap image.

    Returns:
        None
    """
    # Generate the contingency matrix
    contingency_matrix = pd.crosstab(df[col1], df[col2])
    print(f"Contingency matrix between '{col1}' and '{col2}':\n{contingency_matrix}")

    # Drop the "O" category if present
    contingency_matrix.drop(columns="O", errors="ignore", inplace=True)
    contingency_matrix.drop(index="O", errors="ignore", inplace=True)

    # Check if the contingency matrix is empty
    if contingency_matrix.empty:
        print(f"Warning: Contingency matrix for '{col1}' and '{col2}' is empty after processing. Skipping heatmap.")
        return

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        contingency_matrix.T, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True
    )
    plt.title(f"Heatmap of {col1} vs {col2}")
    plt.ylabel(col2)
    plt.xlabel(col1)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    print(f"Heatmap saved to '{output_file}'")
    plt.close()


def main():
    """
    Main function to execute the script.
    """
    # Set the working directory and load the dataset
    txtdir = filesdir
    set_working_directory(txtdir)

    filename = "output.decision"
    try:
        df = load_dataset(filename)
    except FileNotFoundError as e:
        print(e)
        return

    # Generate and save heatmaps for specified column pairs
    heatmap_configs = [
        ("BIO1", "BIO2", "crosstab_pandas_bio1_bio2.png"),
        ("BIO1", "BIO3", "crosstab_pandas_bio1_bio3.png"),
    ]

    for col1, col2, output_file in heatmap_configs:
        try:
            plot_confusion_matrix(df, col1, col2, output_file)
        except Exception as e:
            print(f"Error while generating heatmap for {col1} vs {col2}: {e}")


if __name__ == "__main__":
    main()