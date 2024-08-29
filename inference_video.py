import os
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
pretrained = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")
# fixed config options
#model.get_model().config.ratio = 0.5
model.get_model().config.segment_pruning = False
model.get_model().config.segment_length = 16
model.get_model().config.plot_sys_user_prompt_sim = False
# model.get_model().config.video_name = paths[0].split('/')[-1].removesuffix('.mp4')

# variable config options
model.get_model().config.original_seq_length = -1
model.get_model().config.focus_layers = np.array([3])
model.get_model().config.smooth_forward_segments = np.array([1])
model.get_model().config.focus_llm = False
#get boolean value from string
model.get_model().config.reforward = True
num_frames = 16
if num_frames < 80:
    model.get_model().config.use_cpu = False
    model.get_model().config.use_sequential = False
elif num_frames < 96:
    model.get_model().config.use_cpu = False
    model.get_model().config.use_sequential = True
else:
    model.get_model().config.use_cpu = True
    model.get_model().config.use_sequential = True
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
video_path = "docs/RoadAccidents127_x264.mp4"
video_frames = load_video(video_path, num_frames)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "vicuna_v1"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
#question = f"{DEFAULT_IMAGE_TOKEN}\nWhat happens between the truck and the train?"
"""question = f"{DEFAULT_IMAGE_TOKEN}\nThis video's subtitles are listed below:\nthis has been Messi's tournament he's\n \
Bailey comes to meet him\n\
out the first half of the first half\n\
cold as ice\n\
of Di Maria and then not overcooking it\n\
12 minutes to go he feels the contact\n\
net comes back across goal\n\
and the execution from killing and Bape\n\
defending him back into the middle to\n\
latora Martinez is going to get to that\n\
doesn't matter latora Martinez on side\n\
France will shoot first\n\
yes\n\
they call them one he delivers\n\
Select the best answer to the following multiple-choice question based on the video.\n\
Which significant football event is depicted in the video? \n\
A. FIFA World Cup 2022 Final. \n\
B. 2023 UEFA Champions League final.\n\
C. FIFA World Cup 2018 Final. \n\
D. None of the above. \n\
Respond with only the letter (A, B, C, or D) of the correct option. Option: """

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True,
)
text_outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)
print(text_outputs[0])
