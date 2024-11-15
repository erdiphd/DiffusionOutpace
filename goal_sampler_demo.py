import torch
#from visualize.visualize_2d import *
import argparse
import cv2
import random
import numpy as np
import os

import sys
sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette'))

from infer import single_img
from core.util import tensor2img 


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

IMG_DIMS = 64

parser = argparse.ArgumentParser()
parser.add_argument("-a","--agent", type=str, default = 'expl_agent_800000_sawyer_push.pt')
parser.add_argument("-i", "--image", type = str, default = 'test_images/sawyer_push/heatmap_episode3944.jpg')
parser.add_argument("-i", "--image_yz", type = str, default = 'test_images/sawyer_push/heatmap_episode3944.jpg')
parser.add_argument("-e", "--episode", type = str, default = 'sawyer_peg_push_demo')
parser.add_argument("-env", "--env_name", type = str, default = 'sawyer_peg_push')
parser.add_argument("-dn", "--diffusion_number_of_steps", type = int, default = 250)
args = parser.parse_args()

env_name = args.env_name
LOWER_CONTEXT_BOUNDS = 0
UPPER_CONTEXT_BOUNDS = 0
cur_model_dir = ""

if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
    LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 10])
    cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/PointUMaze/300_Network.pth')
elif env_name in ['sawyer_peg_pick_and_place']:
    LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478])
    UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.4])
    cur_model_dir_xy = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/sawyer_peg_pick_and_place_xy/300_Network.pth')
    cur_model_dir_yz = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/sawyer_peg_pick_and_place_yz/300_Network.pth')
elif env_name ==  'sawyer_peg_push':
    LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
    UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.02])
    cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/sawyer_peg_push/300_Network.pth')
elif env_name == "PointSpiralMaze-v0":
    LOWER_CONTEXT_BOUNDS = np.array([-10, -10]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 10])
    cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/PointSpiralMaze/300_Network.pth')
elif env_name in ["PointNMaze-v0"]:
    LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 18])
    cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'weights/PointNMaze/300_Network.pth')


if args.env not in ['sawyer_peg_pick_and_place']: 
    img = cv2.imread(args.image)
    maze_dim_y = UPPER_CONTEXT_BOUNDS[0] - LOWER_CONTEXT_BOUNDS[0]
    maze_dim_x = UPPER_CONTEXT_BOUNDS[1] - LOWER_CONTEXT_BOUNDS[1]
else:
    img_xy = cv2.imread(args.image)
    img_yz = cv2.imread(args.image_yz)
    maze_dim_y = UPPER_CONTEXT_BOUNDS[0] - LOWER_CONTEXT_BOUNDS[0]
    maze_dim_x = UPPER_CONTEXT_BOUNDS[1] - LOWER_CONTEXT_BOUNDS[1]
    maze_dim_z = UPPER_CONTEXT_BOUNDS[2] - LOWER_CONTEXT_BOUNDS[2]

goal_coordinates = []

while not goal_coordinates:
    if args.env not in ['sawyer_peg_pick_and_place']:
        output_imgs = single_img(img,True, f"./inpaint_results/episode_{args.episode}/", model_pth = cur_model_dir, diffusion_n_steps = args.diffusion_number_of_steps)
        i = 0
        for output_img in output_imgs:
            completed_image = tensor2img(output_img)
            gray_image = cv2.cvtColor(completed_image,cv2.COLOR_BGR2GRAY)
            gray_image = np.flipud(gray_image)
            if np.min(gray_image) < 5:
                goal_coordinate = (np.argwhere(gray_image == np.min(gray_image))[0] / IMG_DIMS)
                goal_coordinate[0] = goal_coordinate[0] * maze_dim_x
                goal_coordinate[1] = goal_coordinate[1] * maze_dim_y
                goal_coordinate = np.flip(goal_coordinate)

                if args.env in ['sawyer_peg_push']:
                    temp_goal = np.zeros(3)
                    temp_goal[0] = goal_coordinate[0]
                    temp_goal[1] = goal_coordinate[1]
                    temp_goal[2] = 0.01478
                    goal_coordinate = temp_goal
                    goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[0]
                    goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[1]
                    if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                        if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0]:
                            goal_coordinate[0] = LOWER_CONTEXT_BOUNDS[0] + 0.002
                        if goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                            goal_coordinate[0] = UPPER_CONTEXT_BOUNDS[0] - 0.002
                    if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                        if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1]:
                            goal_coordinate[1] = LOWER_CONTEXT_BOUNDS[1] + 0.002
                        if goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                            goal_coordinate[1] = UPPER_CONTEXT_BOUNDS[1] - 0.002
                else:
                    goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.5,0.5) + LOWER_CONTEXT_BOUNDS[0]
                    goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.5,0.5) + LOWER_CONTEXT_BOUNDS[1]
                    if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                        if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0]:
                            goal_coordinate[0] = LOWER_CONTEXT_BOUNDS[0] + 0.02
                        if goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                            goal_coordinate[0] = UPPER_CONTEXT_BOUNDS[0] - 0.02
                    if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                        if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1]:
                            goal_coordinate[1] = LOWER_CONTEXT_BOUNDS[1] + 0.02
                        if goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                            goal_coordinate[1] = UPPER_CONTEXT_BOUNDS[1] - 0.02
                goal_coordinates.append(goal_coordinate)
            i += 1
        if not goal_coordinates:
            print("No goals detected, running inference again.")
    else:
        output_imgs_xy = single_img(img_xy,True, f"./inpaint_results_xy/episode_{args.episode}/", model_pth = cur_model_dir_xy, diffusion_n_steps = args.diffusion_number_of_steps)
        output_imgs_yz = single_img(img_yz,True, f"./inpaint_results_yz/episode_{args.episode}/", model_pth = cur_model_dir_yz, diffusion_n_steps = args.diffusion_number_of_steps)
        i = 0
        for output_img_xy in output_imgs_xy:
            completed_image_xy = tensor2img(output_img_xy)
            completed_image_yz = tensor2img(output_imgs_yz[i])
            gray_image_xy = cv2.cvtColor(completed_image_xy,cv2.COLOR_BGR2GRAY)
            gray_image_xy = np.flipud(gray_image_xy)
            gray_image_yz = cv2.cvtColor(completed_image_yz,cv2.COLOR_BGR2GRAY)
            gray_image_yz = np.flipud(gray_image_yz)

            goal_coordinate_xy = (np.argwhere(gray_image_xy == np.min(gray_image_xy))[0] / IMG_DIMS)
            goal_coordinate_xy[0] = goal_coordinate_xy[0] * maze_dim_x
            goal_coordinate_xy[1] = goal_coordinate_xy[1] * maze_dim_y
            goal_coordinate_xy = np.flip(goal_coordinate_xy)

            goal_coordinate_yz = (np.argwhere(gray_image_yz == np.min(gray_image_yz))[0] / IMG_DIMS)
            goal_coordinate_yz[0] = goal_coordinate_yz[0] * maze_dim_y
            goal_coordinate_yz[1] = (goal_coordinate_yz[1] * (maze_dim_z))

            goal_coordinate = np.zeros(3)
            goal_coordinate[0] = goal_coordinate_xy[0]
            goal_coordinate[1] = (goal_coordinate_xy[1] + goal_coordinate_yz[0])/2
            goal_coordinate[2] = goal_coordinate_yz[1]
            goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[0]
            goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[1]
            goal_coordinate[2] = goal_coordinate[2] + random.uniform(-0.01,0.01) + LOWER_CONTEXT_BOUNDS[2]
            if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0]:
                    goal_coordinate[0] = LOWER_CONTEXT_BOUNDS[0] + 0.002
                if goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                    goal_coordinate[0] = UPPER_CONTEXT_BOUNDS[0] - 0.002
            if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1]:
                    goal_coordinate[1] = LOWER_CONTEXT_BOUNDS[1] + 0.002
                if goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                    goal_coordinate[1] = UPPER_CONTEXT_BOUNDS[1] - 0.002
            if goal_coordinate[2] <= LOWER_CONTEXT_BOUNDS[2] or goal_coordinate[2] >= UPPER_CONTEXT_BOUNDS[2]:
                if goal_coordinate[2] <= LOWER_CONTEXT_BOUNDS[2]:
                    goal_coordinate[2] = LOWER_CONTEXT_BOUNDS[2] + 0.002
                if goal_coordinate[2] >= UPPER_CONTEXT_BOUNDS[2]:
                    goal_coordinate[2] = UPPER_CONTEXT_BOUNDS[2] - 0.002
            goal_coordinates.append(goal_coordinate)
            i += 1
        if not goal_coordinates:
            print("No goals detected, running inference again.")
print("Goals are at", goal_coordinates)

agent = torch.load(args.agent)
goal_candidates = np.array(goal_coordinates)
num_grid_point = goal_candidates.shape[0]

if env_name ==  'sawyer_peg_push': 
    initial_states = np.tile(np.array([0.4, 0.8, 0.02]), (num_grid_point, 1))
else:
    initial_states = np.tile(np.array([0,0]), (num_grid_point, 1))

print(initial_states.shape)

observes = torch.as_tensor(np.concatenate([initial_states, goal_candidates], axis= -1), device = 0).float()
print(observes.shape)
aim_output = agent.aim_discriminator.forward(observes).detach().cpu().numpy().flatten()
print("Aim output is ", aim_output)

probabilities = softmax(-1*aim_output).flatten()
print("Probabilities",probabilities)
optimal_goal = np.random.choice(aim_output, p = probabilities)
optimal_goal_index = np.argmax(aim_output==optimal_goal)

print(f"Optimal goal with coordinate {goal_candidates[optimal_goal_index]}")