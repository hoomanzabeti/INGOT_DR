#!/bin/bash


#for DRUG in 'amikacin' 'capreomycin' 'ciprofloxacin' 'ethambutol' 'ethionamide' 'isoniazid' 'kanamycin' 'moxifloxacin' 'ofloxacin' 'pyrazinamide' 'rifampicin' 'streptomycin'
for DRUG in 'ciprofloxacin'

do
  for MODEL in 'INGOT' 'LR_l1' 'LR_l2' 'SVM_l1' 'SVM_l2' 'RF'
  do

    echo ${DRUG}
    echo ${MODEL}
    python INGOT_ML_comparison.py --config=config.yml --data-file=../data/SNPsMatrix_${DRUG}.csv --label-file=../data/${DRUG}Label.csv --drug-name=${DRUG} --model-name=${MODEL}
  done
done