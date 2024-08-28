import copy
import os
import warnings

import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates



def model_init(model_path=None, focus_layers=None, smooth_forward_segments=None, reforward=False, nr_frames=16):
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    model_name = "llava_qwen"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name,
                                                                          device_map=device_map,
                                                                          attn_implementation="eager")
    # fixed config options
    # model.get_model().config.ratio = 0.5
    model.get_model().config.segment_pruning = True
    model.get_model().config.segment_length = 16
    model.get_model().config.plot_sys_user_prompt_sim = False
    # model.get_model().config.video_name = paths[0].split('/')[-1].removesuffix('.mp4')
    # variable config options
    # model.get_model().config.original_seq_length = -1
    model.get_model().config.focus_layers = np.fromstring(focus_layers, sep=',', dtype=int)
    if model.get_model().config.focus_layers[0] == -1:
        model.get_model().config.focus_llm = False
        model.get_model().config.segment_pruning = False
        model.get_model().config.reforward = reforward
    else:
        model.get_model().config.smooth_forward_segments = np.fromstring(smooth_forward_segments, sep=',', dtype=int)
        model.get_model().config.focus_llm = True
        # get boolean value from string
        model.get_model().config.reforward = reforward

    num_frames = nr_frames
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

    model.eval()

    return model, tokenizer, image_processor, "qwen_1_5"


def infer(model, video, instruct, tokenizer, image_processor, do_sample=False, version='llama_2'):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        version (str): conversation template version.
    Returns:
        str: response of the model.
    """

    instruct = DEFAULT_IMAGE_TOKEN + "\n" + instruct
    conv = copy.deepcopy(conv_templates[version])
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image_sizes = [frame.size for frame in video]
    video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
    model.get_model().config.original_seq_length = -1
    # Generate response
    cont = model.generate(
        input_ids,
        images=[video_tensor],
        image_sizes=image_sizes,
        do_sample=do_sample,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)
    return outputs[0]


def x_infer(video, question, model, tokenizer, image_processor, mode='vanilla', do_sample=False, version='qwen_1_5'):
    if mode == 'mcqa':
        instruction = f'{question}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        return infer(model=model, video=video, instruct=instruction, tokenizer=tokenizer, image_processor=image_processor,  do_sample=do_sample, version=version)
    elif mode == 'openend':
        instruction = f'{question}\nAnswer the question using a single word or a short phrase with multiple words.'
        return infer(model=model, video=video, instruct=instruction, tokenizer=tokenizer, image_processor=image_processor,  do_sample=do_sample, version=version)
    elif mode == 'vanilla':
        instruction = question
        return infer(model=model, video=video, instruct=instruction, tokenizer=tokenizer, image_processor=image_processor,  do_sample=do_sample, version=version)
