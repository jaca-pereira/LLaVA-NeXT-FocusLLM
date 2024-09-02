from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")
model.get_model().config.segment_pruning = True
model.get_model().config.segment_length = 16
model.get_model().config.plot_sys_user_prompt_sim = False
# model.get_model().config.video_name = paths[0].split('/')[-1].removesuffix('.mp4')
# variable config options
# model.get_model().config.original_seq_length = -1
model.get_model().config.focus_layers = [-1]
if model.get_model().config.focus_layers[0] == -1:
    model.get_model().config.focus_llm = False
    model.get_model().config.segment_pruning = False
    model.get_model().config.reforward = True
else:
    model.get_model().config.smooth_forward_segments = np.fromstring("1", sep=',', dtype=int)
    model.get_model().config.focus_llm = True
    # get boolean value from string
    model.get_model().config.reforward = True

num_frames = 4
if model.get_model().config.focus_layers[0] == -1:
    model.get_model().config.use_cpu = False
    model.get_model().config.use_sequential = False
else:
    if num_frames < 80:
        model.get_model().config.use_cpu = False
        model.get_model().config.use_sequential = False
    elif num_frames < 96:
        model.get_model().config.use_cpu = False
        model.get_model().config.use_sequential = True
    else:
        model.get_model().config.use_cpu = True
        model.get_model().config.use_sequential = True
model.get_model().config.original_seq_length = -1
model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Load and process video
video_path = "docs/jobs.mp4"
video_frames = load_video(video_path, 16)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]
modalities = ["video"] * len(video_frames)

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=modalities,
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
