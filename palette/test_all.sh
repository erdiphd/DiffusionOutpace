#!/bin/bash

python run.py -p test -c config/inpainting_u_maze_64_test_large_black.json
python run.py -p test -c config/inpainting_u_maze_64_test_small_black.json
python run.py -p test -c config/inpainting_u_maze_64_test_large.json
python run.py -p test -c config/inpainting_u_maze_64_test.json
