import cv2
from ultralytics import YOLO
import numpy as np
import math
import os
import shutil
import pandas as pd
import time
import argparse


# constants
tmp_frames_dir_path = 'data/_tmp/frames'
tmp_crops_dir_path = 'data/_tmp/crops'
tmp_segment_filepath = f'data/_tmp/segment.mp4'
segment_length_sec = 2


# helper functions
def calculate_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_new_bounding_box(x1, y1, x2, y2, w, h, max_width, max_height):
    # make w and h int
    max_width = int(max_width)
    max_height = int(max_height)

    # Calculate the center of the original bounding box
    center_x, center_y = calculate_center(x1, y1, x2, y2)

    # Calculate the new bounding box's coordinates
    new_x1 = math.ceil(center_x - w / 2)
    new_y1 = math.ceil(center_y - h / 2)
    new_x2 = math.ceil(center_x + w / 2)
    new_y2 = math.ceil(center_y + h / 2)

    # Adjust the new bounding box coordinates to ensure they are within the video frame
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = new_x1 + w
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = new_y1 + h
    if new_x2 > max_width:
        new_x2 = max_width
        new_x1 = new_x2 - w
    if new_y2 > max_height:
        new_y2 = max_height
        new_y1 = new_y2 - h

    return [new_x1, new_y1, new_x2, new_y2]


def calculate_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def transform_bbox_for_max_center(tracking_dict, video_width, video_height):

    # 1. create max rectangle for each dog

    max_bbox_whs = dict()
    for sub_id, bboxs in tracking_dict.items():
        max_w, max_h = 0, 0

        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            w, h = abs(x2 - x1), abs(y2 - y1)
            max_w, max_h = max(max_w, w), max(max_h, h)

        max_bbox_whs[sub_id] = (int(max_w) + 10, int(max_h) + 10) # offset by 10 pixels around the subject

    # 2. calculate rectangles depending on the list of bounding boxes

    new_tracking_dict = dict()

    for sub_id, bboxs in tracking_dict.items():
        new_tracking_dict[sub_id] = []

        for bbox in bboxs:
            new_bbox = calculate_new_bounding_box(
                *bbox,
                max_bbox_whs[sub_id][0],
                max_bbox_whs[sub_id][1],
                video_width, video_height
            )
            new_tracking_dict[sub_id].append(new_bbox)

    return new_tracking_dict


# def refresh_tmp_frames_dir():
#     if os.path.exists(tmp_frames_dir_path):
#         shutil.rmtree(tmp_frames_dir_path)
#     os.makedirs(tmp_frames_dir_path, exist_ok=True)


# def refresh_tmp_crops_dir():
#     if os.path.exists(tmp_crops_dir_path):
#         shutil.rmtree(tmp_crops_dir_path)
#     os.makedirs(tmp_crops_dir_path, exist_ok=True)


def refresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def create_tmp_frames_dir_for_subjects(subjects_ids):
    # refresh frame dir
    # refresh_tmp_frames_dir()

    refresh_dir(tmp_frames_dir_path)
    parent_dir = tmp_frames_dir_path

    # create subjects directories
    subjects_ids = [str(sub_id) for sub_id in subjects_ids]
    for directory in subjects_ids:
        os.makedirs(os.path.join(parent_dir, directory), exist_ok=True)


# todo: check fps is specified where called
def create_video(frames_dir, video_location, fps=30, extension='.png'):
    images = sorted([img for img in os.listdir(frames_dir) if img.endswith(extension)])

    # get fame dimensions
    frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _ = frame.shape

    # video writer object
    video = cv2.VideoWriter(
        video_location,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(frames_dir, image)))

    # release the video writer
    video.release()


def select_dog_ids_with_label_frequency(data_dict, label = 16, threshold = 0.3):
    selected_ids = []
    for subject_id, labels in data_dict.items():
        count = labels.count(label)
        if count / len(labels) > threshold:
            selected_ids.append(subject_id)
    return selected_ids


def build_dict(total_frames, all_ids, all_items):
    my_dict = dict()
    for i in range(total_frames):
        for sub_id, item in zip(all_ids[i], all_items[i]):
            if sub_id in my_dict:
                my_dict[sub_id].append(item)
            else:
                my_dict[sub_id] = [item]
    return my_dict


def initial_tracking(video_file_path):

    # reinit the model every time seemns to help - might be because of persist?
    model = YOLO('models/yolov8n.pt')

    # process video
    cap = cv2.VideoCapture(video_file_path)

    all_boxes = []
    all_ids = []
    all_classes = []

    current_frame = 0

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True) # TODO: what is persist?

        all_boxes.append(results[0].boxes.xyxy.tolist())
        all_ids.append(results[0].boxes.id.int().tolist()) # NOTE: here it can error out
        all_classes.append(results[0].boxes.cls.int().tolist())

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    total_frames = current_frame

    # 1. create tracking dictionary
    # dict {sub_id: [bbox1, bbox2 ...] (for each frame where it's recognised)}
    tracking_dict = build_dict(total_frames, all_ids, all_boxes)

    # 2. create class dictionary
    # dict {sub_id: [class1, class2 ...] (for each frame where it's recognised)}
    class_dict = build_dict(total_frames, all_ids, all_classes)

    # 3. get dog ids to filter for dogs below
    dog_ids = select_dog_ids_with_label_frequency(class_dict)

    # 4. filter just the ones which have all frames and are dogs
    tracking_dict = {
        sub_id: bboxs
        for sub_id, bboxs in tracking_dict.items()
        if len(bboxs) == total_frames and sub_id in dog_ids
    }

    return tracking_dict, total_frames


# transform tracking_dict to have the subjects (bounding boxes) you want to crop per frame
# [{sub_id_1: bbox, sub_id_2: bbox}, {sub_id_1: bbox, sub_id_2: bbox} ...] # length = no of frames
def get_subjects_for_each_frame(tracking_dict, total_frames):

    subjects_to_crop = [dict() for _ in range(total_frames)] # a dict for each frame
    for sub_id, bboxs in tracking_dict.items():
        for i, bbox in enumerate(bboxs):
            subjects_to_crop[i][sub_id] = bbox

    return subjects_to_crop


def crop_subjects_and_save_crops(video_file_path, frame_by_frame_subjects):

    # 1. refresh frames and crops directories
    create_tmp_frames_dir_for_subjects(subjects_ids=list(frame_by_frame_subjects[0].keys()))
    # refresh_tmp_crops_dir()
    refresh_dir(tmp_crops_dir_path)

    # 2. write frames
    cap = cv2.VideoCapture(video_file_path)
    current_frame = 0

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        # crop subjects
        for sub_id, box in frame_by_frame_subjects[current_frame].items():
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f'{tmp_frames_dir_path}/{sub_id}/frame-{current_frame:05}.png', crop)

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    # 3. save videos from frames
    subjects_ids = list(frame_by_frame_subjects[0].keys())
    for sub_id in subjects_ids:
        create_video(
            frames_dir=f'{tmp_frames_dir_path}/{sub_id}',
            video_location=f'{tmp_crops_dir_path}/{sub_id:03}.mp4'
        )


def process_segment(input_file_path, video_metadata):
    # 1. get video details
    # video_width, video_height, video_fps = get_video_details(input_file_path)
    video_width, video_height, video_fps = video_metadata

    # 2. get tracking dictionary
    tracking_dict, total_frames = initial_tracking(video_file_path=input_file_path)

    # todo: there's a index out of range somewhere - happens when it can't detect a dog - fix it

    # 3. transform tracking boxes and get subject box for each frame
    tracking_dict = transform_bbox_for_max_center(tracking_dict, video_width, video_height)
    frame_by_frame_subjects = get_subjects_for_each_frame(tracking_dict, total_frames)

    # 4. crop subjects and save them
    crop_subjects_and_save_crops(input_file_path, frame_by_frame_subjects)

    subjects = list(frame_by_frame_subjects[0].keys())
    return subjects


def append_log_exceptions(file_exceptions, file_name, index):
    file = open('logs/error_logs.txt', 'a')
    file.write(f'------------------------------------------------------- \n')
    file.write(f'------------------------------------------------------- \n')
    file.write(f'row number {index:03} | video {file_name} processing \n')
    file.write(f'file exceptions :: \n')
    for i, ex in enumerate(file_exceptions):
        file.write(f'exception {i:03} :: {ex} \n')
    file.close()


def get_video_metadata(cap):
    video_fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, frame_height, frame_width, video_fps


def convert_to_samples_starting_at(start_index):

    df = pd.read_csv('data/raw/labels.csv')
    df = df.iloc[start_index:]  # 746

    success_data = []
    exceptions_data = []

    refresh_dir('data/_tmp')

    for index, row in df.iterrows():

        file_name = row['file_path']
        dataset = row['dataset']
        action = row['action']

        original_video_id = file_name.split('.')[0]

        cap = cv2.VideoCapture(f'data/raw/{file_name}')
        frame_count, frame_height, frame_width, video_fps = get_video_metadata(cap)
        frames_per_segment = int(video_fps * segment_length_sec)

        segment_number = 0
        file_exceptions = []

        # process each segment
        for start_frame in range(0, frame_count, frames_per_segment):

            # 1. create segment
            end_frame = min(start_frame + frames_per_segment + 1, frame_count)
            if end_frame - start_frame < frames_per_segment:
                break

            # set cap to start at segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # create tmp video file for segment
            out = cv2.VideoWriter(tmp_segment_filepath, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

            while cap.isOpened():
                success, frame = cap.read()
                if not success or cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                    break
                out.write(frame)

            out.release()

            try:

                # 2. process segment
                video_metadata = (frame_width, frame_height, video_fps)
                subject_ids = process_segment(tmp_segment_filepath, video_metadata)

                # 3. add prefix and move videos
                for subject_id in subject_ids:
                    original_file = f'{tmp_crops_dir_path}/{subject_id:03}.mp4'
                    final_file_name = f'{original_video_id}_{segment_number:03}_{subject_id:03}.mp4'
                    shutil.copy(original_file, f'data/samples/{final_file_name}')

                    success_data.append([final_file_name, original_video_id, f'{segment_number:03}', dataset, action])
                    # success_data.append([file_name, segment_number, final_file_name, dataset, action])

            except Exception as e:
                exceptions_data.append([original_video_id, f'{segment_number:03}', dataset, action, e])
                file_exceptions.append(e)

            # next segment
            segment_number += 1

        # append logs for each file
        append_log_exceptions(file_exceptions, file_name, index)

        cap.release()

    # save logs dataframes
    success_df = pd.DataFrame(
        success_data,
        columns=['file_path', 'original_file', 'segment', 'dataset', 'action']
    )

    error_df = pd.DataFrame(
        exceptions_data,
        columns=['original_file', 'segment', 'dataset', 'action', 'error']
    )

    success_df.to_csv('logs/all_success_df.csv', index=False)
    error_df.to_csv('logs/all_error_df.csv', index=False)

    # TODO: success_df append to the old labels df
    # labels_df = pd.read_csv('data/samples/labels.csv')
    # labels_df = pd.concat([labels_df, success_df])
    # labels_df.to_csv('data/samples/labels.csv')


parser = argparse.ArgumentParser(description='Process raw and add to samples.')
parser.add_argument('start_index', type=int, help='row number in file_index column of where to start processing.')

args = parser.parse_args()

convert_to_samples_starting_at(args.start_index)
