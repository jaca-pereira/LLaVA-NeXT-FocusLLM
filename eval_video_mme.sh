#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate llava

echo $CONDA_DEFAULT_ENV

cd ~/data/LLaVA-NeXT-FocusLLM

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

focus_layers_list=(
"3" "8" "12" "3" "8" "12" \
"3" "8" "12" "3" "8" "12" \
"3" "8" "12" "3" "8" "12" "3 8" "8 12" \
"3" "8" "12" "3" "8" "12" "3 8" "8 12" \
"3" "8" "12" "3" "8" "12" "3 8" "8 12" "3 8 12" \
"3" "8" "12" "3" "8" "12" "3 8" "8 12" "3 8 12" "3 8 12")

focus_segments_list=(
"1" "1" "1" "1" "1" "1" \
"1" "1" "1" "1" "1" "1" \
"1" "1" "1" "1" "1" "1" "2 1" "2 1" \
"1" "1" "1" "1" "1" "1" "2 1" "2 1" \
"1" "1" "1" "1" "1" "1" "3 1" "3 1" "4 2 1" \
"1" "1" "1" "1" "1" "1" "3 1" "3 1" "4 2 1" "5 3 1")

reforward_list=(
True True True False False False \
True True True False False False \
True True True False False False False False \
True True True False False False False False \
True True True False False False False False False \
True True True False False False False False False False)

nr_frames_list=(
32 32 32 32 32 32 \
48 48 48 48 48 48 \
64 64 64 64 64 64 64 64 \
80 80 80 80 80 80 80 80 \
96 96 96 96 96 96 96 96 96 \
128 128 128 128 128 128 128 128 128 128)

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/video/eval/eval_video_mcqa_videomme.sh --focus_layers "-1" --focus_segments "1" --reforward True --nr_frames 16
mv eval_output/videomme/ eval_output/videomme_replicate

for i in {0..47}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/video/eval/eval_video_mcqa_videomme.sh --focus_layers "${focus_layers_list[i]}" --focus_segments "${focus_segments_list[i]}" --reforward "${reforward_list[i]}" --nr_frames "${nr_frames_list[i]}"
    mv eval_output/videomme/ eval_output/videomme_${focus_layers_list[i]}_${focus_segments_list[i]}_${reforward_list[i]}_${nr_frames_list[i]}
done
