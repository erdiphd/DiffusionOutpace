import torch
from visualize.visualize_2d import *
import cv2
import random
import numpy as np

agent_dir = "expl_agent_30000.pt"

agent = torch.load(agent_dir)

visualize_discriminator(normalizer = None,
                        discriminator = agent.aim_discriminator, 
                        initial_state = np.array([0,0]), 
                        scatter_states = np.array([0,0]),
                        env_name = "PointUMaze-v0", 
                        aim_input_type = 'default',
                        device = 0, 
                        savedir_w_name = 'aim_f_visualize_train_postion_fix_demo',
                        )

visualize_discriminator2(normalizer = None,
                        discriminator = agent.aim_discriminator, 
                        env_name = "PointUMaze-v0", 
                        aim_input_type = 'default',
                        device = 0, 
                        savedir_w_name = "aim_f_visualize_train_goalfix_demo",
                        )
