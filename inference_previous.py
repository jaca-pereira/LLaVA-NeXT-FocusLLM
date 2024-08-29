import argparse
import torch
from matplotlib import pyplot as plt

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
import torch.nn.functional as F
import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu
import seaborn as sns
from transformers import AutoConfig

import cv2
import base64

from PIL import Image

import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default="./docs/jobs.mp4")
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default="./results")
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default="results")
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str,
                        default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default="What happens between the truck and the train?")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--mm_pooling_position", type=str, default="after")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    return parser.parse_args()


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
    if len(frame_idx) >= args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        # replicate frames until they reach the correct number of frames
        frame_idx = np.linspace(0, total_frame_num-1, args.for_get_frames_num, dtype=int).tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # Save frames as images
    # for i, frame in enumerate(spare_frames):
    #     cv2.imwrite(f'{args.output_dir}/frame_{i}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return spare_frames, frame_idx, fps


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def visualize_attention_vectors(attentions, output_ids, tokenizer, modal_token_position, num_image_video_tokens, filename, prompt):

    # Decode the output_ids to tokens
    output_ids = output_ids.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(output_ids)
    tokens = [token.replace('▁', ' ') for token in tokens]

    # Distinguish repeated tokens
    for i in range(1, len(tokens)):
        for j in range(i):
            if tokens[i] == tokens[j]:
                tokens[i] = f'{tokens[i]}_{i}'

    os.makedirs('./figures', exist_ok=True)
    os.makedirs(f'./figures/{filename}', exist_ok=True)

    start_image_video = modal_token_position + 1
    end_image_video = start_image_video + num_image_video_tokens
    end_input = attentions[0][0].shape[-1]
    for idx, attention in enumerate(tqdm(attentions[1:])):
        fig, ax = plt.subplots(figsize=(20, 5))

        # The attention for the current token is a 1D array
        attention_vector = torch.cat(attention, 0).mean(dim=(0, 1))

        # Normalize the attention vector
        norm_attention_vector = F.normalize(attention_vector, p=1, dim=-1).squeeze().cpu().detach().numpy()

        norm_attention_vector = np.clip(norm_attention_vector, 0, 0.1)

        # Plot the attention vector as a bar plot
        ax.bar(range(len(norm_attention_vector)), norm_attention_vector, color='blue')
        ax.set_xticks(range(len(norm_attention_vector)))
        ax.set_title(f'Attention for Token {tokens[idx+1]}')
        ax.set_xlabel('Previous Tokens')
        ax.set_ylabel('Attention')
        # Add vertical dashed lines to delimit the segments
        line_begin_image = ax.axvline(x=start_image_video - 0.5, color='red', linestyle='dashed', linewidth=1, label='begin_image_video')
        line_end_image = ax.axvline(x=end_image_video - 0.5, color='green', linestyle='dashed', linewidth=1, label='end_image_video')
        line_end_input = ax.axvline(x=end_input - 0.5, color='black', linestyle='dashed', linewidth=1, label='end_input')
        plt.tight_layout()
        ax.legend(handles=[line_begin_image, line_end_image, line_end_input], loc='upper center')
        os.makedirs(f'./figures/{filename}/attention_vectors/', exist_ok=True)
        plt.savefig(f'./figures/{filename}/attention_vectors/{prompt}_attention_token_{idx}.png')
        plt.close()


def visualize_hidden_states(all_hidden_states, modal_token_position, num_image_video_tokens, filename, video_path):
    start_image_video = modal_token_position + 1
    end_image_video = start_image_video + num_image_video_tokens
    end_input = all_hidden_states[0][0][0].shape[0]
    hidden_states = torch.cat(all_hidden_states[0], 0)
    #normalize hidden states
    hidden_states = F.normalize(hidden_states, p=2, dim=-1)
    sims = hidden_states @ hidden_states.transpose(-2, -1)
    max_sim = torch.max(sims)
    min_sim = torch.min(sims)
    print(f'Maximum similarity: {max_sim}')
    print(f'Minimum similarity: {min_sim}')
    sims = sims.cpu().detach().numpy()
    os.makedirs('./figures', exist_ok=True)
    os.makedirs(f'./figures/{filename}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/hidden_states', exist_ok=True)
    os.makedirs(f'./figures/{filename}/hidden_states/{video_path}', exist_ok=True)
    for layer_nr, sim in enumerate(sims):
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim, annot=False, cmap='coolwarm', vmin=min_sim, vmax=max_sim)
        plt.axvline(start_image_video + 1, color='green', linestyle='dashed', linewidth=1, label='begin_image_video')
        plt.axvline(end_image_video, color='green', linestyle='dashed', linewidth=1,
                    label='end_image_video')
        plt.axhline(start_image_video, color='green', linestyle='dashed', linewidth=1)
        plt.axhline(end_image_video, color='green', linestyle='dashed', linewidth=1)
        # legend
        plt.legend(loc='upper right')
        plt.title(f'Cosine Similarity Matrix for layer {layer_nr}')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        plt.savefig(f'./figures/{filename}/hidden_states/{video_path}/layer_{layer_nr}.png')
        plt.close()

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """

    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_pooling_position"] = args.mm_pooling_position
        overwrite_config["mm_newline_position"] = args.mm_newline_position

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        # import pdb;pdb.set_trace()

        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
        else:
            least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

        scaling_factor = math.ceil(least_token_number / 4096)
        if scaling_factor >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                print(float(scaling_factor))
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                                   model_name, load_8bit=args.load_8bit,
                                                                                   overwrite_config=overwrite_config, attn_implementation="eager") #IMPORTANT added att_impl
    else:
        pass

    model.get_model().config.ratio = 0.5
    model.get_model().config.focus_layers = np.array([3, 5, 8])
    model.get_model().config.smooth_forward_segments = np.array([8, 4, 1])
    model.get_model().config.focus_llm = False
    model.get_model().config.segment_pruning = False
    model.get_model().config.use_cpu = False
    model.get_model().config.use_sequential = False
    model.get_model().config.plot_sys_user_prompt_sim = False
    # model.get_model().config.video_name = paths[0].split('/')[-1].removesuffix('.mp4')
    model.get_model().config.reforward = False
    model.get_model().config.segment_length = 16

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    video_path = args.video_path

    all_video_pathes = []

    # Check if the video_path is a directory or a file
    if os.path.isdir(video_path):
        # If it's a directory, loop over all files in the directory
        for filename in os.listdir(video_path):
            # Load the video file
            cur_video_path = os.path.join(video_path, f"{filename}")
            all_video_pathes.append(os.path.join(video_path, cur_video_path))
    else:
        # If it's a file, just process the video
        all_video_pathes.append(video_path)

    for video_path in all_video_pathes:

        sample_set = {}
        question = args.prompt
        sample_set["Q"] = question
        sample_set["video_name"] = video_path

        # Check if the video exists
        if os.path.exists(video_path):
            video, _, _ = load_video(video_path, args)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]

        # try:
        # Run inference on the video and add the output to the list

        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).cuda()

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        with torch.inference_mode():
            model.config.output_hidden_states = True
            model.config.output_attentions = True
            outputs_dict = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.1, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True) # stopping_criteria=[stopping_criteria])
            #output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.1, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True) # stopping_criteria=[stopping_criteria])

           # attentions = outputs_dict["attentions"]
            #hidden_states = outputs_dict["hidden_states"]
            output_ids = outputs_dict["sequences"]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        modal_token_position = (input_ids == IMAGE_TOKEN_INDEX).nonzero()[0, 1].item()
        #visualize_attention_vectors(attentions, output_ids, tokenizer, modal_token_position,
        #                           model.model.num_image_video_features, video_path.split('/')[-1].removesuffix('.mp4'),
        #                            qs)
        #visualize_hidden_states(hidden_states, modal_token_position, model.get_model().num_image_video_features, "llava-next", video_path.split('/')[-1].removesuffix('.mp4'))
        print(f"Question: {prompt}\n")
        print(f"Response: {outputs}\n")


        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        sample_set["pred"] = outputs
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
