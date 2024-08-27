#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llava

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaVA-NeXT-FocusLLM

python3 inference_video.py



#OUTPUT_DIR=eval_output
#CKPT_NAME=llava-onevision-qwen2-0.5b-ov
#output_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/merge.json
#output_sub_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/merge_sub.json
#outputs_files="$output_file,$output_sub_file"

#python llava/eval/eval_video_mcqa_videomme.py \
#    --results_file $outputs_files \
#    --video_duration_type "short,medium,long" \
#    --focus_layers "3" \
#    --focus_segments "1" \
#    --selection_type True \
#    --nr_frames 32 \
#    --skip_missing
