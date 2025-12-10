# microrobot-dl
## Local environment setup
Using pip
```
pip install -e .
```
Using uv
```
uv sync
```

## Running the model
### Data structure
The data need to be in the following structure for both train/test.<br>
See the [training source](https://huggingface.co/datasets/Lan-2025/OpticalMicrorobot/blob/main/Deep_Learning_2025_Dataset/README.md) for details.
```
.
└── P{i}_R{j}/
│   ├── P{i}_R{j}_depth.txt
│   ├── {anyname1}.jpg
│   ├── {anyname2}.jpg
└── …
```

## Taining 
Run the notebook `./notebooks/training.ipynb`. <br>
It supposes the data is in `./data/2025_Dataset`.<br>
If running on colab, it will clone the repo automatically.

## Testing
Run the notebook `notebooks/test.ipynb`.
Please config the path of data in the first cell.<br>
If running on colab, it will clone the repo automatically.