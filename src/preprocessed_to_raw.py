import pandas as pd
import argparse
import os
import json
import shutil


def add_to_raw(dataset_path, dataset_name):

    # functionality description:
    # * looks into data/raw
    # * checks what the last index is for the raw data
    # * starts the unique_index from there - to use in order to add the new dataset
    # * gets the path, action dictionary in order to add files to raw
    # * converts everything and adds the files to the previous labels dataframe in raw

    # 1. init current index, either start from scratch or continue from raw

    data_columns = ['file_index', 'file_path', 'dataset', 'action', 'original_file_path']

    df = pd.DataFrame(columns=data_columns)
    current_index = 0
    print("Initialising current index to 0.")

    raw_labels_path = 'data/raw/labels.csv'
    if os.path.exists(raw_labels_path):
        df = pd.read_csv(raw_labels_path)
        current_index = df.iloc[-1]['file_index'] + 1
        print(f"Found existing files in raw, overwriting current index to {current_index}.")

    # 2. read in path action dictionary for preprocessed dataset

    with open(f'{dataset_path}/path_action_dict.json', 'r') as f:
        path_action_dict = json.load(f)

    # 3. go over the files and add them to raw

    data = []
    for file_path, action_label in path_action_dict.items():
        new_file_name = f'{current_index:06}.mp4'

        # raw data dataframe format:
        # index | video_name (6 digit unique id) | dataset name | label | original video path
        data.append([current_index, new_file_name, dataset_name, action_label, file_path])
        shutil.copy(file_path, f'data/raw/{new_file_name}')
        current_index += 1

    new_df = pd.DataFrame(data, columns=data_columns)

    # overwrite dataframe with new df
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(raw_labels_path, index=False)


# define command-line arguments
parser = argparse.ArgumentParser(description='Add preprocessed dataset to raw')
parser.add_argument('preprocessed_dataset_path', type=str, help='Path to the preprocessed dataset to add to raw data.')
parser.add_argument('dataset_name', type=str, help='Name of the dataset in order to use in the raw labels csv')

args = parser.parse_args()

add_to_raw(args.preprocessed_dataset_path, args.dataset_name)
