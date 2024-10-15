import torch
#from visualize.visualize_2d import *
import argparse
import cv2
import random
import numpy as np

import sys
sys.path.insert(1, '/home/faruk/Palette-Image-to-Image-Diffusion-Models')

import infer
from core.util import tensor2img


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#MAZE_DIMS = 12
#MAZE_MIN = -2
#MAZE_MAX = 10
IMG_DIMS = 64

#image_dir = "./test_images/GT_heatmap_episode248.jpg"
#agent_dir = "expl_agent_40000.pt"
#episode = "999"

parser = argparse.ArgumentParser()
parser.add_argument("-a","--agent", type=str, default = 'expl_agent_800000_sawyer_push.pt')
parser.add_argument("-i", "--image", type = str, default = 'test_images/sawyer_push/heatmap_episode3944.jpg')
parser.add_argument("-e", "--episode", type = str, default = 'sawyer_peg_push_demo')
parser.add_argument("-env", "--env_name", type = str, default = 'sawyer_peg_push')
parser.add_argument("-bn", "--beta_number_of_steps", type = int, default = 250)
args = parser.parse_args()

env_name = args.env_name
LOWER_CONTEXT_BOUNDS = 0
UPPER_CONTEXT_BOUNDS = 0
cur_model_dir = ""

if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
    LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 10])
    cur_model_dir = "/home/faruk/Palette-Image-to-Image-Diffusion-Models/experiments/inpainting_u_maze_64_path_oriented/checkpoint/300_Network.pth"
elif env_name in ['sawyer_peg_pick_and_place']:
    LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
    UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.4])            
elif env_name ==  'sawyer_peg_push':
    LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
    UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.02])
    cur_model_dir = "/home/faruk/Palette-Image-to-Image-Diffusion-Models/experiments/inpainting_sawyer_push_path_oriented/checkpoint/300_Network.pth"
elif env_name == "PointSpiralMaze-v0":
    LOWER_CONTEXT_BOUNDS = np.array([-10, -10]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 10])
    cur_model_dir = "/home/faruk/Palette-Image-to-Image-Diffusion-Models/experiments/inpainting_spiral_maze_64_path_oriented/checkpoint/300_Network.pth"
elif env_name in ["PointNMaze-v0"]:
    LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
    UPPER_CONTEXT_BOUNDS = np.array([10, 18])
    cur_model_dir = "/home/faruk/Palette-Image-to-Image-Diffusion-Models/experiments/inpainting_n_maze_64_path_oriented/checkpoint/300_Network.pth"


img = cv2.imread(args.image)

#img = cv2.imread(image_dir)


maze_dim_y = UPPER_CONTEXT_BOUNDS[0] - LOWER_CONTEXT_BOUNDS[0]
maze_dim_x = UPPER_CONTEXT_BOUNDS[1] - LOWER_CONTEXT_BOUNDS[1]

goal_coordinates = []

while not goal_coordinates:
    output_imgs = infer.single_img(img,True, f"./inpaint_results/episode_{args.episode}/", model_pth = cur_model_dir,beta_schedule_n_steps = args.beta_number_of_steps)
    for output_img in output_imgs:
        completed_image = tensor2img(output_img)
        gray_image = cv2.cvtColor(completed_image,cv2.COLOR_BGR2GRAY)
        gray_image = np.flipud(gray_image)
        if np.min(gray_image) < 5:
            goal_coordinate = (np.argwhere(gray_image == np.min(gray_image))[0] / IMG_DIMS)
            goal_coordinate[0] = goal_coordinate[0] * maze_dim_x
            goal_coordinate[1] = goal_coordinate[1] * maze_dim_y
            goal_coordinate = np.flip(goal_coordinate)

            if env_name ==  'sawyer_peg_push':
                temp_goal = np.zeros(3)
                temp_goal[0] = goal_coordinate[0]
                temp_goal[1] = goal_coordinate[1]
                temp_goal[2] = 0.01478
                goal_coordinate = temp_goal

            print(goal_coordinate)
            if env_name ==  'sawyer_peg_push':
                goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[0]
                goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.05,0.05) + LOWER_CONTEXT_BOUNDS[1]
                if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                    if goal_coordinate[0] <= LOWER_CONTEXT_BOUNDS[0]:
                        goal_coordinate[0] = LOWER_CONTEXT_BOUNDS[0] + 0.002
                    if goal_coordinate[0] >= UPPER_CONTEXT_BOUNDS[0]:
                        goal_coordinate[0] = UPPER_CONTEXT_BOUNDS[0] - 0.002
                if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[0]:
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
                if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[0]:
                    if goal_coordinate[1] <= LOWER_CONTEXT_BOUNDS[1]:
                        goal_coordinate[1] = LOWER_CONTEXT_BOUNDS[1] + 0.02
                    if goal_coordinate[1] >= UPPER_CONTEXT_BOUNDS[1]:
                        goal_coordinate[1] = UPPER_CONTEXT_BOUNDS[1] - 0.02
            goal_coordinates.append(goal_coordinate)
    if not goal_coordinates:
        print("No goals detected, running inference again.")
print("Goals are at", goal_coordinates)

agent = torch.load(args.agent)
#agent = torch.load(agent_dir)

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

print(f"Optimal goal is output{optimal_goal_index}.jpg with coordinate {goal_candidates[optimal_goal_index]}")

#np.random.choice 
'''
def softmax(x):
e_x = np.exp(x - np.max(x))
return e_x / e_x.sum()

# Input values
x = -1 * np.array([[-0.9845953 ],
[-0.9791703 ],
[-1.4541411 ],
[-0.83480513],
[-1.713158 ],
[-0.8628219 ],
[-0.8927 ],
[-1.8103262 ]])

# Calculate the softmax probabilities
probabilities = softmax(x)
'''

#Possibly use q as well for goal selection
#outpace agent.critic --> Q
