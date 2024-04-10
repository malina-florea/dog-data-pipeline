# Dog Video Processing


## Example Runs

## Step 1: preprocess

```shell
python preprocess_dataset.py a2d_preprocess data/datasets/dataset_1 
```

```shell
python preprocess_dataset.py drive_preprocess data/datasets/dataset_2 
```

## Step 2: preprocessed to raw
```shell
python preprocessed_to_raw.py data/preprocessed/dataset_1 a2d_online   
```

```shell
 python preprocessed_to_raw.py data/preprocessed/dataset_2 drive_1
```

```shell
python preprocessed_to_raw.py data/preprocessed/dataset_3 drive_2
```

## Step 3: raw to samples
```shell
python raw_to_samples.py 472
```