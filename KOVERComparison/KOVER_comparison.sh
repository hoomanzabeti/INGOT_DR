#!/bin/bash

FOLD=5
SEED=33
MAXRULESIZE=20
DATADIR=../../data/

mkdir kover_report
for DRUG in 'amikacin' 'capreomycin' 'ciprofloxacin' 'ethambutol' 'ethionamide' 'isoniazid' 'kanamycin' 'moxifloxacin' 'ofloxacin' 'pyrazinamide' 'rifampicin' 'streptomycin'

do
  echo ${DRUG}
  mkdir -p "${DRUG}"
  cd ${DRUG}
  kover dataset create from-tsv --genomic-data ${DATADIR}SNPsMatrix_${DRUG}.tsv --phenotype-description ${DRUG}" resistance"  --phenotype-metadata ${DATADIR}${DRUG}Metadata.tsv --output ${DRUG}.kover --progress
  kover dataset split --dataset ${DRUG}.kover --id ${DRUG}_${SEED}_split --train-ids ${DATADIR}${DRUG}_train_isolates.txt --test-ids ${DATADIR}${DRUG}_test_isolates.txt --folds ${FOLD} --random-seed ${SEED} --progress
  kover learn scm --dataset ${DRUG}.kover --split ${DRUG}_${SEED}_split --model-type conjunction disjunction --p 0.1 1.0 10.0 100.0 1000.0 --max-rules ${MAXRULESIZE} --hp-choice cv --n-cpu 32 --progress
  mv report.txt ${DRUG}_report.txt
  cp ${DRUG}_report.txt ../koverReport/${DRUG}_report.txt
  cd ..
done