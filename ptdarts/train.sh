#!/usr/bin/env bash
cat train.sh
cat environment.yaml
nvidia-smi
#conda update -n base -c defaults conda
conda env create -f environment.yaml
#conda init bash
#exec bash
source activate lfm
nvidia-smi
nvidia-smi
python search.py --batch_size 128

