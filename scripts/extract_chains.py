import os
from Bio.PDB import PDBParser

def extract_chain(pdb_file):
    # extracts chain of pdb
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    chain_ids = set()

    for model in structure:
        for chain in model:
            chain_ids.add(chain.get_id())
    
    if len(chain_ids) == 1:
        return chain_ids.pop()  
    else:
        return None

def create_chain_mapping(pdb_folder, mapping_file):
    # iterates over all pdbs in the folder
    mapping_lines = []
    
    for filename in os.listdir(pdb_folder):
        if filename.lower().endswith('.pdb'):
            pdb_path = os.path.join(pdb_folder, filename)
            chain = extract_chain(pdb_path)
            if chain is None:
                print(f"Ignoring {filename}: more than one chain found.")
            else:
                mapping_lines.append(f"{filename} {chain}")
    
    with open(mapping_file, 'w') as f:
        for line in mapping_lines:
            f.write(line + "\n")
    print(f"Mapping file '{mapping_file}' created with {len(mapping_lines)} entries.")

if __name__ == "__main__":
    pdb_folder = "training_pdbs"     
    mapping_file = "chain_mapping.txt" 
    
    create_chain_mapping(pdb_folder, mapping_file)
