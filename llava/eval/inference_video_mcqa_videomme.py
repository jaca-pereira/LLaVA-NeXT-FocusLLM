import os
import re
import math
import json
import copy
import argparse
import warnings
import traceback
from platform import processor

import pysubs2
import numpy as np
import pyarrow.parquet as pq
from decord import VideoReader, cpu
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import sys

from llava.eval import x_infer, model_init
from llava.model.builder import load_pretrained_model

#sys.path.append('./')

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_idx, fps   # (frames, height, width, channels)


class VideoMMEDataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, subtitle_folder, data_list, processor, max_num_frames=16):
        self.video_folder = video_folder
        self.subtitle_folder = subtitle_folder
        self.data_list = data_list
        self.processor = processor
        self.max_num_frames = max_num_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]

        video_ytid = line['url'].split('watch?v=')[-1]

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.video_folder, f'{video_ytid}{fmt}')
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        subtitle_path = os.path.join(self.subtitle_folder, f'{video_ytid}.srt')

        try:
            video_tensor, selected_frame_ids, fps = load_video(video_path, self.max_num_frames)
            num_frames = video_tensor.shape[0]
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_ytid}')
            video_tensor = None
            num_frames = 0

        if video_tensor is not None and os.path.exists(subtitle_path):
            subs = pysubs2.load(subtitle_path, encoding="utf-8")
            subtitles = []
            for seleced_frame_id in selected_frame_ids:
                sub_text = ""
                cur_time = pysubs2.make_time(fps=fps, frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = "\n".join(subtitles)
        else:
            subtitles = ""

        return {
            'video': video_tensor,
            'subtitle': subtitles,
            'record': line,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    sub = [x['subtitle'] for x in batch]
    rcs = [x['record'] for x in batch]
    return vid, sub, rcs


def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    # Convert PyArrow Table to pandas DataFrame
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append({
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "question": record.question,
                        "choices": list(record.options),
                        "answer": record.answer,
                    }
                ]
            })
        else:
            jsons[-1]['questions'].append({
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            })

    return jsons


def build_videomme_eval(args, processor):
    # convert parquet to json
    questions = load_parquet(args.question_file)
    # questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoMMEDataset(args.video_folder, args.subtitle_folder, questions, processor, args.nr_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    return dataloader


def videomme_dump(record, instruct, output):
    letters = ['A', 'B', 'C', 'D']

    pred_answer = re.findall('[\(\ \[]*([A-D])[\)\.\ \]]*', output)
    try:
        assert len(pred_answer) >= 1, 'The video \"{}\" output \"{}\" is not in the expected format'.format(
            record['youtube_id'], instruct + '\n' + output)
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip('()')
        pred_idx = letters.index(pred_answer)
    except:
        traceback.print_exc()
        pred_idx = 2

    return letters[pred_idx]


def run_inference(args):
    # Initialize the model
    model, tokenizer, image_processor, version = model_init(args.model_path, args.focus_layers, args.focus_segments, args.reforward, args.nr_frames)

    answer_file = os.path.expanduser(args.answer_file)
    answer_sub_file = answer_file.replace('.json', '_sub.json')
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    ans_sub_file = open(answer_sub_file, "w")

    val_loader = build_videomme_eval(args, image_processor)
    print(f"Start inference on {len(val_loader)} samples")

    # Iterate over each sample in the ground truth file
    for i, (videos, subtitles, records) in enumerate(tqdm(val_loader)):
        video_tensor = videos[0]
        subtitle = subtitles[0]
        record = records[0]

        new_record = copy.deepcopy(record)
        new_record_sub = copy.deepcopy(record)

        if video_tensor is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            new_record_sub['missing'] = True
            ans_sub_file.write(json.dumps(new_record_sub) + ",\n")
            continue
        else:
            new_record['missing'] = False
            new_record_sub['missing'] = False

        questions = record['questions']
        for idx, question in enumerate(questions):
            q = question['question']
            ops = question['choices']

            instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            instruct += f"{q}\n"
            for op_idx, op in enumerate(ops):
                instruct += f"{op}\n"
            instruct += "The best answer is: "
            output = x_infer(video_tensor, instruct, mode='vanilla', model=model, tokenizer=tokenizer, image_processor=image_processor,do_sample=False, version="vicuna_v1")
            new_record['questions'][idx]['response'] = videomme_dump(record, instruct, output)

            instruct = f"This video's subtitles are listed below:\n{subtitle}\n" + instruct
            output = x_infer(video_tensor, instruct, mode='vanilla', model=model, tokenizer=tokenizer, image_processor=image_processor, do_sample=False, version="vicuna_v1")
            new_record_sub['questions'][idx]['response'] = videomme_dump(record, instruct, output)

        ans_file.write(json.dumps(new_record) + ",\n")
        ans_sub_file.write(json.dumps(new_record_sub) + ",\n")

    ans_file.close()
    ans_sub_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--subtitle-folder', help='Directory containing subtitle files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--focus_layers', help='Focus layers for the model.', type=str)
    parser.add_argument('--focus_segments', help='Focus segments for the model.', type=str)
    parser.add_argument('--reforward', help='Reforward parameter for the model.', type=bool)
    parser.add_argument('--nr_frames', help='Number of frames to process.', type=int)
    parser.add_argument('--version', help='Conv template', type=str)

    args = parser.parse_args()
    #mp.set_start_method('spawn')
    run_inference(args)
