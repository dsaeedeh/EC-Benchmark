import argparse
from Bio import SeqIO

def merge_fastq_files(fastq_files):
    fqs = [SeqIO.parse(f, "fastq") for f in fastq_files]
    while True:
        for fq in fqs:
            try:
                print(next(fq).format("fastq"), end="")
            except StopIteration:
                fqs.remove(fq)
        if len(fqs) == 0:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple FastQ files')
    parser.add_argument('fastq_files', nargs='+', type=str, help='List of FastQ files to merge')
    args = parser.parse_args()

    merge_fastq_files(args.fastq_files)
