import argparse
import urllib.request
import os
import tarfile
import shutil

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and extract UniProt files')
parser.add_argument('year1', type=str, help='Year for testing Swissprot')
parser.add_argument('month1', type=str, help='Month for testing Swissprot')
parser.add_argument('year2', type=str, help='Year for pretraining and training')
parser.add_argument('month2', type=str, help='Month for pretraining and training')
args = parser.parse_args()

year1 = args.year1
month1 = args.month1
year2 = args.year2
month2 = args.month2

# Set output directory
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Download testing Swissprot
testing_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year1}_{month1}/knowledgebase/uniprot_sprot-only{year1}_{month1}.tar.gz"
urllib.request.urlretrieve(testing_url, output_dir + f"/uniprot_sprot-only{year1}_{month1}.tar.gz")

# Download pretraining and training
pretraining_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year2}_{month2}/knowledgebase/knowledgebase{year2}_{month2}.tar.gz"
urllib.request.urlretrieve(pretraining_url, output_dir + f"/knowledgebase{year2}_{month2}.tar.gz")

# Extract uniprot_sprot
tar_file = output_dir + f"/uniprot_sprot-only{year1}_{month1}.tar.gz"
with tarfile.open(tar_file, "r:gz") as tar:
    tar.extract("uniprot_sprot.dat.gz")
os.rename("uniprot_sprot.dat.gz", f"uniprot_sprot{year1}_{month1}.data.gz")
shutil.move(f"uniprot_sprot{year1}_{month1}.data.gz", output_dir + f"/uniprot_sprot{year1}_{month1}.data.gz")

# Extract knowledgebase
tar_file = output_dir + f"/knowledgebase{year2}_{month2}.tar.gz"
with tarfile.open(tar_file, "r:gz") as tar:
    tar.extract("uniprot_sprot.dat.gz")
    tar.extract("uniprot_trembl.dat.gz")
os.rename("uniprot_sprot.dat.gz", f"uniprot_sprot{year2}_{month2}.data.gz")
shutil.move(f"uniprot_sprot{year2}_{month2}.data.gz", output_dir + f"/uniprot_sprot{year2}_{month2}.data.gz")
os.rename("uniprot_trembl.dat.gz", f"uniprot_trembl{year2}_{month2}.data.gz")
shutil.move(f"uniprot_trembl{year2}_{month2}.data.gz", output_dir + f"/uniprot_trembl{year2}_{month2}.data.gz")

# Download alphafold structures (PDB files) for SwissprotKB:
pdb_url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar"
urllib.request.urlretrieve(pdb_url, output_dir + f"/swissprot_pdb_v4.tar")

# Extract alphafold structures (PDB files) for SwissprotKB:
tar_file = output_dir + f"/swissprot_pdb_v4.tar"
with tarfile.open(tar_file, "r:") as tar:
    tar.extract("swissprot_pdb_v4")
shutil.move("swissprot_pdb_v4/", output_dir + "/swissprot_pdb_v4")


