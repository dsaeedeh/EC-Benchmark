mkdir data

$1: 2022
$2: 05
$3: 2018
$4: 02

# Download testing Swissprot $1 $2
curl https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$1_$2/knowledgebase/uniprot_sprot-only$1_$2.tar.gz -o uniprot_sprot-only$1_$2.tar.gz

# Download pretraining and training $3 $4
curl https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$3_$4/knowledgebase/knowledgebase$3_$4.tar.gz -o knowledgebase$3_$4.tar.gz

# Swissprot - test
tar -xf uniprot_sprot-only$1_$2.tar.gz uniprot_sprot.xml.gz
gunzip uniprot_sprot.xml.gz
mv uniprot_sprot.xml uniprot_sprot_$1_$2.xml 

# pretraining and training
tar -xf knowledgebase$3_$4.tar.gz 
gunzip uniprot_trembl$3_$4.dat.gz
mv uniprot_sprot.xml uniprot_sprot_$1_$2.xml