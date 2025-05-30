#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import argparse

def compute_distance_matrix(points):
    # compute differences using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return distance_matrix

def process_csv_file(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return
    
    required_columns = {'X', 'Y', 'Z'}
    if not required_columns.issubset(df.columns):
        print(f"Skipping {input_path}: missing one or more required columns {required_columns}.")
        return
    
    points = df[['X', 'Y', 'Z']].values
    if points.shape[0] == 0:
        print(f"Skipping {input_path}: no data found.")
        return
    
    distance_matrix = compute_distance_matrix(points)
    pd.DataFrame(distance_matrix).to_csv(output_path, index=False)
    print(f"Processed {input_path} -> {output_path}")

def process_all_csvs(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + "_distance_matrix.csv"
            output_path = os.path.join(output_folder, output_filename)
            process_csv_file(input_path, output_path)

if __name__ == "__main__":
    process_all_csvs("3d_coords", "training_distmatrices")
