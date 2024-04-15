
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data-gym` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python3 download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
# dc 
python3 experiment.py --env hopper --dataset medium --model_type dc --K 8 --embed_dim 256 --learning_rate 0.0001

# dt
python3 experiment.py --env hopper --dataset medium --model_type dt --embed_dim 256 --learning_rate 0.0001
```

Adding `-w True` will log results to Weights and Biases.
