#!/bin/bash
set -eu  # Crash if variable used without being set

# We use the mbr-nmt repository
git clone git@github.com:Roxot/mbr-nmt.git
cd mbr-nmt
pip install .

# Configure params
utility=bleu
n=50

# Format the input files
raw_file_cascaded=$1
raw_file_e2e=$2
samples_file_combined=$3

python format_utilities.py --function "format-from-csv" --file-paths ${raw_file_cascaded} 
python format_utilities.py --function "format-from-csv" --file-paths ${raw_file_e2e} 

# Run MBR
echo "MBR on cascaded"
samples_file_cascaded=${raw_file_cascaded%.*}.hyps
output_file=${samples_file_cascaded%.*}.mbr_${utility}.${samples_file_cascaded##*.}

mbr-nmt translate -s ${samples_file_cascaded} -n ${n} -u ${utility} -o ${output_file} 
mbr-nmt convert -f mbr-nmt -i ${output_file} -o ${output_file}



echo "MBR on E2E"
samples_file_e2e=${raw_file_e2e%.*}.hyps
output_file=${samples_file_e2e%.*}.mbr_${utility}.${samples_file_e2e##*.}

mbr-nmt translate -s ${samples_file_e2e} -n ${n} -u ${utility} -o ${output_file} 
mbr-nmt convert -f mbr-nmt -i ${output_file} -o ${output_file}


echo "MBR on combined"
python format_utilities.py --function "combine-hyps" --file-paths "${samples_file_cascaded}|${samples_file_e2e}" --nr-samples "50|50" --out-combined ${samples_file_combined}
output_file=${samples_file_combined%.*}.mbr_${utility}.${samples_file_combined##*.}

n=100

mbr-nmt translate -s ${samples_file_combined} -n ${n} -u ${utility} -o ${output_file} ${bleurt_checkpoint_var} # -t 1
mbr-nmt convert -f mbr-nmt -i ${output_file} -o ${output_file}


