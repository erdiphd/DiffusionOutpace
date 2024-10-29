#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50 seed=1
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50 seed=2
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50 seed=3
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50 seed=4
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 aim_discriminator_cfg.lambda_coef=50 seed=5
