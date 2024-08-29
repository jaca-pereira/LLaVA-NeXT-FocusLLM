#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llava

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaVA-NeXT-FocusLLM

python3 MLVU.evaluation_test.test_bench.py
