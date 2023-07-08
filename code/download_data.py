import argparse
import subprocess

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

# Change directory to data
subprocess.run(["cd", "data"])

# Download testing Swissprot
testing_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year1}_{month1}/knowledgebase/uniprot_sprot-only{year1}_{month1}.tar.gz"
subprocess.run(["curl", testing_url, "-o", f"uniprot_sprot-only{year1}_{month1}.tar.gz"])

# Download pretraining and training
pretraining_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year2}_{month2}/knowledgebase/knowledgebase{year2}_{month2}.tar.gz"
subprocess.run(["curl", pretraining_url, "-o", f"knowledgebase{year2}_{month2}.tar.gz"])

# Extract testing Swissprot
subprocess.run(["tar", "-zxvf", f"uniprot_sprot-only{year1}_{month1}.tar.gz"])
subprocess.run(["mv", "uniprot_sprot.dat", f"uniprot_sprot{year1}_{month1}.data"])
subprocess.run(["rm", "-f", f"uniprot_sprot.fasta.gz", f"uniprot_sprot_varsplic.fasta.gz", f"uniprot_sprot.xml.gz"])

# Extract pretraining and training
subprocess.run(["tar", "-zxvf", f"knowledgebase{year2}_{month2}.tar.gz"])
subprocess.run(["mv", "uniprot_sprot.dat.gz", f"uniprot_sprot{year2}_{month2}.data.gz"])
subprocess.run(["mv", "uniprot_trembl.dat.gz", f"uniprot_trembl{year2}_{month2}.data.gz"])
