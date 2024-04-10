import argparse
import importlib
import os
import shutil
from moviepy.editor import VideoFileClip
import json
import pandas as pd


def get_path_action_dict(input_path):

    valid_extensions = ['.mp4', '.mov']

    folder_to_action_mapping = {
        'dogs_eating': 'eat',
        'dogs_playing': 'play',
        'dogs_sleeping': 'sleep',
        'dogs_walking': 'walk'
    }

    path_action_dict = dict()

    for action_folder, action in folder_to_action_mapping.items():
        action_folder_path = f'{input_path}/{action_folder}'
        files = os.listdir(action_folder_path)
        files = [file for file in files if os.path.splitext(file)[1] in valid_extensions]

        for file in files:
            path_action_dict[f'{action_folder_path}/{file}'] = action

    return path_action_dict


def create_dir_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


# TODO: refactor this
def convert_to_mp4(path_action_dict, destination_path):

    for file_path, _ in path_action_dict.items():

        path_components = file_path.split('/')
        old_file_name = path_components[4]

        new_file_dir = f'{destination_path}/{path_components[2]}/{path_components[3]}'
        new_file_name = f"{old_file_name.split('.')[0]}.mp4"
        new_file_path = f'{new_file_dir}/{new_file_name}'

        # create exceptions path for when failing to convert
        create_dir_if_does_not_exist(f'{new_file_dir}/exceptions')

        # if file already converted, skip
        if os.path.isfile(new_file_path):
            continue

        # if mov, convert
        if file_path.endswith('.mov'):

            try:
                clip = VideoFileClip(file_path)
                clip.write_videofile(new_file_path, codec='libx264')

            except Exception as ex:

                exception_file_path = f"{new_file_dir}/exceptions/{old_file_name}"
                shutil.copy(file_path, exception_file_path)
                print(f'exception file: {file_path}')
                print(ex)

        # if mp4, move
        if file_path.endswith('.mp4'):
            shutil.copy(file_path, new_file_path)


def write_dict_to_json(my_dict, dict_path):
    with open(dict_path, "w") as file:
        json.dump(my_dict, file)


def drive_preprocess(input_path, output_path='data/preprocessed'):

    dataset_name = input_path.split('/')[-1]

    path_action_dict = get_path_action_dict(input_path)
    convert_to_mp4(path_action_dict, output_path)

    preprocessed_path_action_dict = get_path_action_dict(f'{output_path}/{dataset_name}')
    write_dict_to_json(preprocessed_path_action_dict, f'{output_path}/{dataset_name}/path_action_dict.json')


def a2d_preprocess(input_path, output_path='data/preprocessed'):

    dataset_name = input_path.split('/')[-1]

    # read in labels data
    relevant_actor_action_labels = [71, 72, 73, 74, 75, 76, 77, 78]  # 79 is none in terms of action
    col_names = ['youtube_id', 'action_label', 'start_ts', 'end_ts', 'heigh', 'width', 'no_frames', 'no_annotated_frames', 'usage']
    df = pd.read_csv(f'{input_path}/videoset.csv', names=col_names, header=None)
    df = df[df['action_label'].isin(relevant_actor_action_labels)]

    label_action_mapping = {
        72: 'play',
        73: 'eat',
        75: 'play',
        76: 'play',
        77: 'run',
        78: 'walk',
        # 79: 'none'
    }

    df = df.replace({"action_label": label_action_mapping})

    # preprocess data and create and save path_action dict
    # todo: can refactor this

    create_dir_if_does_not_exist(f'{output_path}/{dataset_name}/videos')
    preprocessed_path_action_dict = dict()

    for index, row in df.iterrows():
        video_id = row['youtube_id']
        action = row['action_label']

        file_path = f'{input_path}/clips320H/{video_id}.mp4'
        new_file_path = f'{output_path}/{dataset_name}/videos/{video_id}.mp4'

        # copy video and add to dict
        shutil.copy(file_path, new_file_path)
        preprocessed_path_action_dict[new_file_path] = action

    write_dict_to_json(preprocessed_path_action_dict, f'{output_path}/{dataset_name}/path_action_dict.json')


# define command-line arguments
parser = argparse.ArgumentParser(description='Preprocess a dataset.')
parser.add_argument('function_name', type=str, help='The name of the function to call.')
parser.add_argument('input_path', type=str, help='Path to the input dataset.')

args = parser.parse_args()

# call specified function
function = globals()[args.function_name]
function(args.input_path)
