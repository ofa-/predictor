#!/bin/bash

set -x

DATA_URL=https://www.data.gouv.fr/fr/datasets/r
DATA_SET=6fadff46-9efd-4c53-942a-54aca783c30c

curl -sL $DATA_URL/$DATA_SET > data.csv

sed -i '/91;2020-09-18/ d' data.csv


[ -f dep.csv ] || curl -s \
https://www.regions-et-departements.fr/fichiers/departements-francais.csv \
| sed '101,$ d' | sed '2,10 s/^/0/' > dep.csv
