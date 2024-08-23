import os
import re
import json
import argparse
import csv
from typing import List, Dict, Optional, Union

CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def eval_your_results(
        your_results_path: str,
        video_types: Optional[Union[List[str], str]] = None,
        skip_missing: Optional[bool] = True,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "response",
        focus_layers: Optional[str] = None,
        focus_segments: Optional[str] = None,
        selection_type: Optional[str] = None,
        nr_frames: Optional[int] = None
):

    # Load your results
    your_results_path = your_results_path.split(",")
    print(f"Loading your results from {your_results_path}")
    your_results = []
    for file in your_results_path:
        with open(file, 'r') as f:
            your_results.append(json.load(f))

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    row = []
    for i in range(len(your_results)):
        q_type_dict = {}
        overall_results = {}
        for video_type in video_types:
            # Initialize dictionary for each video type with and without subtitles
            q_type_dict[video_type] = {"w/o": {"correct": 0, "answered": 0}}

            # Filter results by duration and subtitle presence
            your_results_video_type_wo = [item for item in your_results[i] if item["duration"] == video_type and not item.get("subtitles", False)]

            subtitle_type, results = "w/o", your_results_video_type_wo
            if not skip_missing:
                assert len(results) == 300, f"Number of files in {video_type} with subtitles {subtitle_type} is not 300. Check if there are missing files."

            for item in results:
                if skip_missing and item.get("missing", False):
                    continue

                questions = item["questions"]

                for question in questions:

                    # Get the ground truth and your response
                    gt_answer = question[gt_answer_key]
                    response = question[your_answer_key]

                    # Extract the answer from the response
                    extraction = extract_characters_regex(response)

                    if extraction != "":
                        q_type_dict[video_type][subtitle_type]["answered"] += 1
                        q_type_dict[video_type][subtitle_type]["correct"] += extraction == gt_answer

            total_correct_wo = q_type_dict[video_type]["w/o"]["correct"]
            total_answered_wo = q_type_dict[video_type]["w/o"]["answered"]

            overall_results[video_type] = {
                    "w/o": 100 * total_correct_wo / total_answered_wo if total_answered_wo > 0 else 0
            }

        # Calculate overall performance for the entire dataset
        total_correct_wo = sum([q_type_dict[video_type]["w/o"]["correct"] for video_type in video_types])
        total_answered_wo = sum([q_type_dict[video_type]["w/o"]["answered"] for video_type in video_types])

        overall_results["Overall"] = {
            "w/o": 100 * total_correct_wo / total_answered_wo if total_answered_wo > 0 else 0
        }

        if i == 0:
            row = [
                overall_results["Overall"]["w/o"],overall_results.get("short", {}).get("w/o", 0),
                overall_results.get("medium", {}).get("w/o", 0), overall_results.get("long", {}).get("w/o", 0)
            ]
            print(f"Results without subtitles: {row}")
        else:
            row = [
                row[0], overall_results["Overall"]["w/o"], row[1], overall_results.get("short", {}).get("w/o", 0),
                row[2], overall_results.get("medium", {}).get("w/o", 0), row[3], overall_results.get("long", {}).get("w/o", 0)
            ]
            print(f"Results with subtitles: {row}")

    # Save results to CSV
    output_dir = "eval_output"
    output_file = os.path.join(output_dir, "video_mme_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.isfile(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write header only if file does not exist
        if not file_exists:
            header = ["# Frames", "Focus Layers", "Focus Segments", "Selection Type", "Overall w/o", "Overall w", "Short w/o", "Short w", "Medium w/o", "Medium w", "Long w/o", "Long w"]
            csvwriter.writerow(header)
        # Write the new row of results
        if focus_layers is not None:
            row = [
                nr_frames, focus_layers, focus_segments, selection_type, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
            ]
        else:
            row = [
                nr_frames, "N/A", "N/A", "N/A", row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
            ]
        print(f"Row to be written: {row}")
        csvwriter.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--video_duration_type", type=str, required=True)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--focus_layers", type=str, required=True)
    parser.add_argument("--focus_segments", type=str, required=True)
    parser.add_argument("--selection_type", type=str, required=True)
    parser.add_argument("--nr_frames", type=int, required=True)

    args = parser.parse_args()

    eval_your_results(
        args.results_file,
        video_types=args.video_duration_type,
        skip_missing=args.skip_missing,
        focus_layers=args.focus_layers,
        focus_segments=args.focus_segments,
        selection_type=args.selection_type,
        nr_frames=args.nr_frames
    )
