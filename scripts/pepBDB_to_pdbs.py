#!/usr/bin/env python3
import os
import shutil

source_folder = "pepbdb"
destination_folder = "training_pdbs"

def extract_and_rename_peptide_files(source_folder, destination_folder):
    if not os.path.isdir(source_folder):
        print(f"Source folder '{source_folder}' does not exist or is not a directory.")
        return

    os.makedirs(destination_folder, exist_ok=True)

    files_found = 0
    files_copied = 0

    # Walk through the directory tree of the source folder
    for root, dirs, files in os.walk(source_folder):
        if "peptide.pdb" in files:
            source_file = os.path.join(root, "peptide.pdb")
            subfolder_name = os.path.basename(root)
            destination_file = os.path.join(destination_folder, f"{subfolder_name}.pdb")
            try:
                shutil.copy(source_file, destination_file)
                print(f"Copied and renamed '{source_file}' to '{destination_file}'.")
                files_copied += 1
            except Exception as e:
                print(f"Error copying '{source_file}': {e}")
            files_found += 1

    print(f"Found {files_found} file(s) named 'peptide.pdb'.")
    print(f"Successfully copied and renamed {files_copied} file(s) to '{destination_folder}'.")

if __name__ == "__main__":
    extract_and_rename_peptide_files(source_folder, destination_folder)
