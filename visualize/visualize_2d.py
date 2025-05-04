

import copy
import pickle as pkl

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.cm as cm
import os
plt.switch_backend('agg')
import seaborn as sns

class UMaze(object):
    def __init__(self) -> None:
        self.ox, self.oy = [], []
        for i in range(-2, 10):
            self.ox.append(i)
            self.oy.append(-2)
        for i in range(-2, 11):
            self.ox.append(10.0)
            self.oy.append(i)
        for i in range(-2, 10):
            self.ox.append(i)
            self.oy.append(10.0)
        for i in range(-2, 10):
            self.ox.append(-2)
            self.oy.append(i)

        for i in range(-2, 6):
            self.ox.append(i)
            self.oy.append(2)
        for i in range(2, 7):
            self.ox.append(6.0)
            self.oy.append(i)
        for i in range(-2, 6):
            self.ox.append(i)
            self.oy.append(6)


class NMaze(object):
    def __init__(self) -> None:
        self.ox, self.oy = [], []
        for i in range(-2, 11):
            self.ox.append(i)
            self.oy.append(-2)
        for i in range(-2, 19):
            self.ox.append(10)
            self.oy.append(i)
        for i in range(-2, 11):
            self.ox.append(i)
            self.oy.append(18)
        for i in range(-2, 19):
            self.ox.append(-2)
            self.oy.append(i)
        for i in range(-2, 7):
            self.ox.append(i)
            self.oy.append(2)
        for i in range(2, 7):
            self.ox.append(6)
            self.oy.append(i)
        for i in range(-2, 7):
            self.ox.append(i)
            self.oy.append(6)
        for i in range(2, 11):
            self.ox.append(i)
            self.oy.append(10)
        for i in range(10, 15):
            self.ox.append(2)
            self.oy.append(i)
        for i in range(2, 11):
            self.ox.append(i)
            self.oy.append(14)


class SpiralMaze(object):
    def __init__(self) -> None:
        self.ox, self.oy = [], []
        for i in range(-2, 10):
            self.ox.append(i)
            self.oy.append(-2)
        for i in range(-2, 7):
            self.ox.append(i)
            self.oy.append(2)
        for i in range(-2, 11):
            self.ox.append(10.0)
            self.oy.append(i)
        for i in range(2, 7):
            self.ox.append(6.0)
            self.oy.append(i)
        for i in range(-10, 11):
            self.ox.append(i)
            self.oy.append(10.0)
        for i in range(-6, 7):
            self.ox.append(i)
            self.oy.append(6.0)
        for i in range(-10, 11):
            self.ox.append(-10.0)
            self.oy.append(i)
        for i in range(-6, 7):
            self.ox.append(-6.0)
            self.oy.append(i)
        for i in range(-10, 11):
            self.ox.append(-10.0)
            self.oy.append(i)
        for i in range(-10, 11):
            self.ox.append(i)
            self.oy.append(-10.0)
        for i in range(-6, 11):
            self.ox.append(i)
            self.oy.append(-6.0)
        for i in range(-10, 11):
            self.ox.append(10.0)
            self.oy.append(i)
        for i in range(-2, 3):
            self.ox.append(-2.0)
            self.oy.append(i)



# visualize along the grid cell in 2D topview





def aim_visualize(class_instance,initial_state, agent, savedir_w_name):
    disc_vis_start_time = time.time()

    num_test_points = 60  # 30
    if class_instance.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif class_instance.cfg.env in ['sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif class_instance.cfg.env == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif class_instance.cfg.env in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None

    if class_instance.cfg.env == "PointSpiralMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointSpiralMaze_background.png')
        world_coordinate = np.float32([[8, -8], [-8, -8], [-8, 8], [8, 8]])
        image_coordinates = np.float32([[395, 395], [85, 395], [85, 85], [395, 85]])
        map = SpiralMaze()
    elif class_instance.cfg.env == "PointUMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointUMaze_background.png')
        world_coordinate = np.float32([[8, 8], [0, 8], [0, 0], [8, 0]])
        image_coordinates = np.float32([[355, 127], [124, 123], [124, 357], [355, 355]])
        map = UMaze()
    elif class_instance.cfg.env == "PointNMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointNMaze_background.png')
        world_coordinate = np.float32([[8, 16], [0, 16], [0, 0], [8, 0]])
        image_coordinates = np.float32([[317, 85], [162, 85], [162, 395], [316, 394]])
        map = NMaze()
    elif class_instance.cfg.env == "PointLongCorridor-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointLongCorridor_background.png')
        world_coordinate = np.float32([[0, 0], [0, 12], [24, 12], [24, 0]])
        image_coordinates = np.float32([[85, 305], [85, 150], [395, 150], [395, 305]])
    else:
        raise NotImplementedError


    grid_x, grid_y = np.meshgrid(x, y)
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])],
                             axis=1)  # [num_test_points^2, 2]

    if class_instance.cfg.env in ['sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457 * np.ones([goal_xy.shape[0], 1])], axis=-1)  # [num_test_points^2, 3]

    num_grid_point = goal_xy.shape[0]

    initial_states = np.tile(initial_state, (num_grid_point, 1))  # [num_test_points^2, dim]

    M = cv2.getPerspectiveTransform(world_coordinate, image_coordinates)
    background_image = mpimg.imread(background_image_path)
    positions = np.stack((x, y), axis=1)
    pixel_positions = cv2.perspectiveTransform(positions.reshape(-1, 1, 2), M).reshape(-1, 2)


    with torch.no_grad():
        observes = torch.as_tensor(np.concatenate([initial_states, goal_xy], axis=-1),
                                   device=class_instance.device).float()  # [num_test_points^2, dim*2]
        if class_instance.cfg.normalize_rl_obs is not None:
            observes = agent.normalize_obs(observes, class_instance.cfg.env)
        aim_outputs = agent.aim_discriminator.forward(observes).detach().cpu().numpy()  # [num_test_points^2, 1]

    v_min, v_max = aim_outputs.min(), aim_outputs.max()

    aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])

    fig, ax = plt.subplots()
    #ax.plot(map.ox, map.oy, ".k")

    c = ax.contourf(pixel_positions[:, 0], pixel_positions[:, 1], aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max,alpha=0.6)
    fig.colorbar(c, ax=ax)
    ax.imshow(background_image)
    if initial_state.ndim == 1:
        ax.scatter(initial_state[0], initial_state[1], marker="*", c='black', s=10, label='Current_position')
    else:
        for t in range(initial_state.shape[0]):
            ax.scatter(initial_state[t, 0], initial_state[t, 1], marker="*", c=str(1. - t / initial_state.shape[0]),
                       s=10, label='s_' + str(t))
    if class_instance.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])
        ax.plot(obstacle_point_x, obstacle_point_y, c='black')
    elif class_instance.cfg.env == "PointSpiralMaze-v0":
        obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
        obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])
        ax.plot(obstacle_point_x, obstacle_point_y, c='black')
    elif class_instance.cfg.env in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])
        ax.plot(obstacle_point_x, obstacle_point_y, c='black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])
        ax.plot(obstacle_point_x, obstacle_point_y, c='black')
    else:
        pass


    ax.axis('off')
    ax.set_aspect('equal', 'box')

    fig.tight_layout()
    fig.savefig(f'{savedir_w_name}_ep_init_fixed_no_back.png', bbox_inches='tight')
    plt.close(fig)


def Q_visualize(class_instance,observe_array, action_array, agent, savedir_w_name):
    number_of_points = 100
    x_debug = torch.linspace(class_instance.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0],
                             class_instance.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0], number_of_points)
    y_debug = torch.linspace(class_instance.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1],
                             class_instance.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1], number_of_points)
    X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
    x_input_debug = X_debug.reshape(-1, 1)
    y_input_debug = Y_debug.reshape(-1, 1)
    virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug)).to(class_instance.device)
    if class_instance.cfg.env == "PointSpiralMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointSpiralMaze_background.png')
        world_coordinate = np.float32([[8, -8], [-8, -8], [-8, 8], [8, 8]])
        image_coordinates = np.float32([[395, 395], [85, 395], [85, 85], [395, 85]])
        map = SpiralMaze()
    elif class_instance.cfg.env == "PointUMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointUMaze_background.png')
        world_coordinate = np.float32([[8, 8], [0, 8], [0, 0], [8, 0]])
        image_coordinates = np.float32([[355, 127], [124, 123], [124, 357], [355, 355]])
        map = UMaze()
    elif class_instance.cfg.env == "PointNMaze-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointNMaze_background.png')
        world_coordinate = np.float32([[8, 16], [0, 16], [0, 0], [8, 0]])
        image_coordinates = np.float32([[317, 85], [162, 85], [162, 395], [316, 394]])
        map = NMaze()
    elif class_instance.cfg.env == "PointLongCorridor-v0":
        background_image_path = os.path.join(os.path.dirname(__file__), 'PointLongCorridor_background.png')
        world_coordinate = np.float32([[0, 0], [0, 12], [24, 12], [24, 0]])
        image_coordinates = np.float32([[85, 305], [85, 150], [395, 150], [395, 305]])
    else:
        raise NotImplementedError

    repeated_state_debug = torch.tile(torch.tensor(observe_array).to(class_instance.device).to(dtype=torch.float32), [len(virtual_diffusion_goals_debug) // observe_array.shape[0], 1])
    repeated_state_debug[:, -2:] = virtual_diffusion_goals_debug
    if class_instance.cfg.normalize_rl_obs:
        repeated_state_debug = agent.normalize_obs(repeated_state_debug, class_instance.cfg.env)

    repeated_action_debug = torch.tile(action_array, [len(virtual_diffusion_goals_debug) // observe_array.shape[0], 1])
    critic_value_debug1, critic_value_debug2 = agent.critic(repeated_state_debug, repeated_action_debug)
    critic_value_debug = (critic_value_debug1 + critic_value_debug2)/2
    critic_value_surface = np.array(critic_value_debug.detach().cpu()).reshape(X_debug.shape)
    critic_value_surface = critic_value_surface.T
    del critic_value_debug1
    del critic_value_debug2
    del repeated_action_debug
    del observe_array

    M = cv2.getPerspectiveTransform(world_coordinate, image_coordinates)
    background_image = mpimg.imread(background_image_path)
    positions = np.stack((x_debug, y_debug), axis=1)
    pixel_positions = cv2.perspectiveTransform(positions.reshape(-1, 1, 2), M).reshape(-1, 2)
    fig1, ax1 = plt.subplots()
    contourf = ax1.contourf(pixel_positions[:, 0], pixel_positions[:, 1], critic_value_surface, cmap='jet', alpha=0.6, levels=100)
    cbar = plt.colorbar(contourf, ax=ax1)
    ax1.imshow(background_image)
    ax1.axis('off')
    ax1.set_aspect('equal', 'box')
    fig1.tight_layout()
    fig1.savefig(f'{savedir_w_name}_ep_Q.png', bbox_inches='tight')
    plt.close(fig1)
    fig2, ax2 = plt.subplots()
    contourf = ax2.contourf(positions[:, 0], positions[:, 1], critic_value_surface, cmap='jet', alpha=0.6, levels=100)
    # ax2.imshow(background_image)
    ax2.plot(map.ox, map.oy, ".k")
    ax2.axis('off')
    ax2.set_aspect('equal', 'box')
    fig2.tight_layout()
    fig2.savefig(f'{savedir_w_name}_ep_Q_no_back.png', bbox_inches='tight')
    plt.close(fig2)



def visualize_discriminator(normalizer, discriminator, initial_state, scatter_states, env_name, aim_input_type, device, savedir_w_name):
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    
    num_test_points = 60 # 30
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in ['sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None
    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
    
    if env_name in [ 'sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]
    
    num_grid_point = goal_xy.shape[0]
    
    initial_states  = np.tile(initial_state, (num_grid_point, 1)) # [num_test_points^2, dim]
    
    with torch.no_grad():
        observes = torch.as_tensor(np.concatenate([initial_states, goal_xy], axis= -1), device = device).float()# [num_test_points^2, dim*2]
        if normalizer is not None:
            observes = normalizer(observes, env_name)
        aim_outputs = discriminator.forward(observes).detach().cpu().numpy() #[num_test_points^2, 1]
    
    v_min, v_max = aim_outputs.min(), aim_outputs.max()           
    
    aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    
    if scatter_states.ndim==1:
        ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
    else:
        for t in range(scatter_states.shape[0]):
            ax.scatter(scatter_states[t, 0], scatter_states[t, 1], marker="*", c = str(1.-t/scatter_states.shape[0]) , s=10, label='s_'+str(t))
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":
        obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
        obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('aim_discriminator_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))
    
def visualize_discriminator2(normalizer, discriminator, env_name, aim_input_type, device, savedir_w_name, goal = None):   
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    num_test_points = 60 # 30
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in [ 'sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None

    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]

    if env_name in [ 'sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]

    num_grid_point = goal_xy.shape[0]
    
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obs_desired_goal = np.array([0., 8.]) 
    elif env_name in [ 'sawyer_peg_push']:
        obs_desired_goal = np.array([-0.3, 0.4, 0.02])
    elif env_name == "PointSpiralMaze-v0":
        obs_desired_goal = np.array([8., -8.]) 
    elif env_name in ["PointNMaze-v0"]:
        obs_desired_goal = np.array([8., 16.]) 

    initial_states  = np.tile(np.array(obs_desired_goal), (num_grid_point, 1)) # [num_test_points^2, dim]
    
    with torch.no_grad():
        observes = torch.as_tensor(np.concatenate([goal_xy, initial_states], axis= -1), device = device).float()# [num_test_points^2, dim*2]
        if normalizer is not None:
            observes = normalizer(observes, env_name)
        aim_outputs = discriminator.forward(observes).detach().cpu().numpy() #[num_test_points^2, 1]
    
    v_min, v_max = aim_outputs.min(), aim_outputs.max()           
    
    aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    # ax.scatter(goal[0], goal[1], marker="*", c = 'black', s=10, label='goal_position')
    ax.scatter(obs_desired_goal[0], obs_desired_goal[1], marker="*", c = 'black', s=10, label='goal_position')
    
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":
        obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
        obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('aim_discriminator_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    plt.close('all')
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))

# visualize along the grid cell in 2D topview    
def visualize_uncertainty(vf, vf_obs_achieved_goal, scatter_states, env_name, aim_input_type, device, savedir_w_name):    
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
        num_test_points = 30
        if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            x = np.linspace(-2, 10, num_test_points)
            y = np.linspace(-2, 10, num_test_points)
        elif env_name == "PointSpiralMaze-v0":
            x = np.linspace(-10, 10, num_test_points)
            y = np.linspace(-10, 10, num_test_points)
        elif env_name in ["PointNMaze-v0"]:
            x = np.linspace(-2, 10, num_test_points)
            y = np.linspace(-2, 18, num_test_points)
        grid_x, grid_y = np.meshgrid(x,y)    
        goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
        num_grid_point = goal_xy.shape[0]
        
        vf_obs_achieved_goals  = np.tile(vf_obs_achieved_goal, (num_grid_point, 1)) # [num_test_points^2, dim]
        
        with torch.no_grad():
            observes = torch.as_tensor(np.concatenate([vf_obs_achieved_goals, goal_xy], axis= -1), device = device).float()# [num_test_points^2, dim*2]
            aim_outputs = vf.std(observes).detach().cpu().numpy() #[num_test_points^2, 1]
        
        v_min, v_max = aim_outputs.min(), aim_outputs.max()           
        
        aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
        
        fig, ax = plt.subplots()

        c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
        
        if scatter_states.ndim==1:
            ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
        else:
            raise NotImplementedError

        if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            obstacle_point_x = np.array([-2, 6, 6, -2])
            obstacle_point_y = np.array([2, 2, 6, 6])  
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        elif env_name == "PointSpiralMaze-v0":         
            obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
            obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        elif env_name in ["PointNMaze-v0"]:
            obstacle_point_x = np.array([-2, 6, 6, -2])
            obstacle_point_y = np.array([2, 2, 6, 6])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
            obstacle_point_x = np.array([10, 2, 2, 10])
            obstacle_point_y = np.array([10, 10, 14, 14])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        

        ax.set_title('uncertainty_visualize')        
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        fig.colorbar(c, ax=ax)
        ax.axis('tight')
        plt.legend(loc="best")
        plt.savefig(savedir_w_name+'.jpg')
        plt.close()

    else:
        raise NotImplementedError    
    disc_vis_end_time = time.time()
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))    
    

# visualize along the grid cell in 2D topview    
def visualize_meta_nml(agent, meta_nml_epoch, scatter_states, replay_buffer, goal_env, env_name, aim_input_type, savedir_w_name):    
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    
    num_test_points = 60
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in ['sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None

    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
    
    if env_name in ['sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]

    observes =goal_xy #torch.as_tensor(goal_xy).float()# [num_test_points^2, dim*2]        
    
    outputs = agent.get_prob_by_meta_nml(observes, meta_nml_epoch, replay_buffer=replay_buffer, goal_env=goal_env) # input : [1, dim] output : list of [dim(1)]
    
    v_min, v_max = outputs.min(), outputs.max()           
    
    outputs = np.reshape(outputs, [num_test_points, num_test_points])
    
    use_smoothing=True
    if use_smoothing:
        from scipy.ndimage import gaussian_filter
        outputs = gaussian_filter(outputs, sigma=2)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    
    if scatter_states.ndim==1:
        ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
    else:
        for t in range(scatter_states.shape[0]):
            ax.scatter(scatter_states[t, 0], scatter_states[t, 1], marker="*", c = str(1.-t/scatter_states.shape[0]) , s=30, label='s_'+str(t))
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":         
            obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
            obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('meta_nml_prob_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    # print('meta nml prob visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))

