"""
PDB Coordinate Extraction

This script extracts the coordinates of specific atoms ('N', 'CA', 'C', and 'O') from PDB files and saves them in a JSON file.

Usage:
    python pdb_coordinate_extraction.py /path/to/pdb_folder --json_dir /path/to/output_json_folder

Arguments:
    pdb_folder : str
        The path to the folder containing PDB files.

Options:
    --json_dir : str (default: current directory)
        The path to the folder where the output JSON file will be saved.

Output:
    A JSON file named 'swissprot_coordinates.json' containing the extracted coordinates.
    The file will be saved in the specified JSON directory or the current directory if not provided.

"""

import os
import json
import argparse

def extract_coordinates(pdb_dir, json_file):
    """
    Extracts coordinates of specified atoms from PDB files and saves them in a JSON file.

    Parameters:
        pdb_dir (str): Path to the folder containing PDB files.
        json_dir (str): Path to the output JSON folder.

    Returns:
        None
    """

    final_dict = {}
    for root, dirs, files in os.walk(pdb_dir):
        for dir in dirs:
            for root_main, dirs_main, files_main in os.walk(os.path.join(pdb_dir, dir)):
                for file in files_main:
                    if file.endswith('.pdb'):
                        protein_id = file.split('-')[1]
                        with open(os.path.join(root_main, file)) as f:
                            lines = f.readlines()
                        coordinates = {'N': [], 'CA': [], 'C': [], 'O': []}
                        for line in lines:
                            if line.startswith('ATOM'):
                                atom_name = line[12:16].strip()
                                if atom_name in ['N', 'CA', 'C', 'O']:
                                    x = float(line[30:38].strip())
                                    y = float(line[38:46].strip())
                                    z = float(line[46:54].strip())
                                    coordinates[atom_name].append([x, y, z])
                        final_dict[protein_id] = coordinates
    with open(json_file, 'w') as f:
        json.dump(final_dict, f)
    print(f"Generated {json_file} with {len(final_dict)} entries")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDB Coordinate Extraction')
    parser.add_argument('pdb_dir', type=str, help='Path to the PDB folder')
    parser.add_argument('--json_file', type=str, default='swissprot_coordinates.json', help='Name of the output JSON file')
    args = parser.parse_args()

    extract_coordinates(args.pdb_dir, args.json_file)

