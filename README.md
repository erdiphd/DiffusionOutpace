# DiffusionOutpace

This is an implementation of OUTPACE with a diffusion based inpainting (Palette) for curriculum goal generation. It is implemented as a master thesis for TUM. Below are the setup instructions for our method, which is the same OUTPACE. Palette also uses the same conda environment with outpace, so there is no need to install anything additional. The method is split into three steps, each will be detailed on how to replicate after setup instructions.

## Setup Instructions
0. Create a conda environment, pip is seperated to track the installed packages, can be commented back in outpace.yml:
```
conda env create -f outpace.yml
conda activate outpace
pip install -r outpace_requirements.txt
```

1. Add the necessary paths:
```
conda develop meta-nml
```

2. Install subfolder dependencies:
```
cd meta-nml && pip install -r requirements.txt
cd ..
chmod +x install.sh
./install.sh
```
3. Install [pytorch](https://pytorch.org/get-started/locally/)


4. Set config_path:
see config/paths/template.yaml

5. To run robot arm environment install [metaworld](https://github.com/rlworkgroup/metaworld):
```
git clone https://github.com/Farama-Foundation/Metaworld.git
git reset --hard 84bda2c
pip install -e .
```
6. a. Create a new directory called "weights" and create 6 subdirectories under it, called "PointUMaze","PointNMaze","PointSpiralMaze","sawyer_peg_push","sawyer_peg_pick_and_place_xy","sawyer_peg_pick_and_place_yz". Put the palette weights (300_Network.pth) for every weight to their respective directory.

    b. Alternatively, the weights that are used during this thesis are uploaded [here](https://drive.google.com/file/d/1-NZP3ivtMJnOrA00uEOr3jzbrScEcP1W/view). Simply unzip it and arrange it to the folder structure described above.


## Usage
### Data Collection for Diffusion Model and Diffusion Based Curriculum Learning

Update the config/config_outpace.yaml for desired method (data collection or diffusion currriculum learning). The related flags are commented in the config file. For reference, the current configuration is for training Point-U Maze with diffusion goal candidate generation and AIM goal selection.

PointUMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointUMaze-v0 aim_disc_replay_buffer_capacity=10000 adam_eps=0.01
```
PointNMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointNMaze-v0 aim_disc_replay_buffer_capacity=10000 adam_eps=0.01
```
PointSpiralMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50
```
sawyer_peg_pick_and_place
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=sawyer_peg_pick_and_place aim_disc_replay_buffer_capacity=30000 normalize_nml_obs=true normalize_f_obs=false normalize_rl_obs=false adam_eps=0.01
```
sawyer_peg_push
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=sawyer_peg_push aim_disc_replay_buffer_capacity=30000 normalize_nml_obs=true normalize_f_obs=false normalize_rl_obs=false adam_eps=0.01 hgg_kwargs.match_sampler_kwargs.hgg_L=0.5
```

For the training of the diffusion model, refer to the README.md in the folder "palette".

The evaluation results (the .csv files) can be found under "thesis results".

# Acknowledgements

Our code is sourced and modified from the official implementation of [OUTPACE](https://github.com/jayLEE0301/outpace_official) and the unofficial implementation of [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models). Also, [mujoco-maze](https://github.com/kngwyu/mujoco-maze) and [metaworld](https://github.com/Farama-Foundation/Metaworld) are used to create the environments.