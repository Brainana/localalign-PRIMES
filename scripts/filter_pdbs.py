#!/usr/bin/env python3
import os

pdb_folder = "training_pdbs"

def remove_non_peptide_pdbs(folder):
    pdb_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdb")]
    
    for pdb_file in pdb_files:
        file_path = os.path.join(folder, pdb_file)
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            remove_file = False
            found_ca = False
            found_non_ca = False

            for line in lines:
                if line.startswith("HETATM"):
                    remove_file = True
                    print(f"Flagging '{pdb_file}' for removal: contains HETATM records.")
                    break

                if line.startswith("ATOM"):
                    tokens = line.split()
                    if len(tokens) < 3:
                        continue

                    atom_name = tokens[2]
                    
                    if atom_name != "CA":
                        found_non_ca = True
                    else:
                        found_ca = True

            if remove_file:
                os.remove(file_path)
                print(f"Removed '{pdb_file}' because it contains HETATM or unknown residue entries.")
            else:
                if found_ca and not found_non_ca:
                    os.remove(file_path)
                    print(f"Removed '{pdb_file}' because it contains only alpha carbon (CA) records.")
                else:
                    print(f"Kept '{pdb_file}'.")
                    
        except Exception as e:
            print(f"Error processing '{pdb_file}': {e}")

if __name__ == "__main__":
    remove_non_peptide_pdbs(pdb_folder)
