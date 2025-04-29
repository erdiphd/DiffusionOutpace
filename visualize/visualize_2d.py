

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





def aim_visualize(class_instance,goal_candidates, agent, savedir_w_name):
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


    for goal_index, selected_goal in enumerate(goal_candidates):

        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()

        repeated_state_debug = torch.tile(torch.tensor(selected_goal).to(class_instance.device),
                                          [virtual_diffusion_goals_debug.shape[0], 1])
        initial_state = np.zeros_like(selected_goal)
        repeated_initial_state = torch.tile(torch.tensor(initial_state).to(class_instance.device),[virtual_diffusion_goals_debug.shape[0], 1])

        inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, repeated_state_debug))
        inputs_norm_tensor_tmp_initial = torch.hstack((repeated_initial_state, virtual_diffusion_goals_debug))

        if class_instance.cfg.normalize_rl_obs:
            inputs_norm_tensor_tmp = agent.normalize_obs(inputs_norm_tensor_tmp, class_instance.cfg.env)
            inputs_norm_tensor_tmp_initial = agent.normalize_obs(inputs_norm_tensor_tmp_initial, class_instance.cfg.env)

        inputs_norm_tensor_tmp = inputs_norm_tensor_tmp.to(dtype=torch.float32)
        inputs_norm_tensor_tmp_initial = inputs_norm_tensor_tmp_initial.to(dtype=torch.float32)

        aim_output_visualize = agent.aim_discriminator.forward(inputs_norm_tensor_tmp).detach().cpu().numpy().flatten()

        M = cv2.getPerspectiveTransform(world_coordinate, image_coordinates)
        background_image = mpimg.imread(background_image_path)
        positions = np.stack((x_debug, y_debug), axis=1)
        pixel_positions = cv2.perspectiveTransform(positions.reshape(-1, 1, 2), M).reshape(-1, 2)
        X_debug_pixel, Y_debug_pixel = np.meshgrid(pixel_positions[:, 0], pixel_positions[:, 1])
        aim_outputs_tmp_pixel = cv2.perspectiveTransform(aim_output_visualize.reshape(-1, 1, 2), M)


        c = ax.pcolormesh(pixel_positions[:, 0], pixel_positions[:, 1],
                          aim_output_visualize.reshape(x_debug.shape[0], -1).T,
                          cmap='RdBu_r', alpha=0.8)
        selected_goal_pixel = cv2.perspectiveTransform(selected_goal.reshape(-1, 1, 2), M).reshape(-1, 2).flatten()
        ax.scatter(selected_goal_pixel[0], selected_goal_pixel[1], marker="*", c='black', s=15, label='goal')
        ax.imshow(background_image)
        ax.axis('off')
        ax.set_aspect('equal', 'box')
        # I don't want to show the pixel values of the aim in the colorbar. therefore the following line is necessary. (but dont visualize it in first graph therefore alpha-
        fig.colorbar(c, ax=ax)
        fig.tight_layout()
        fig.savefig(f'{savedir_w_name}_ep_goal_fixed{goal_index}.png',bbox_inches='tight')
        plt.close(fig)

        c2 = ax3.pcolormesh(x_debug, y_debug, aim_output_visualize.reshape(x_debug.shape[0], -1).T,cmap='RdBu_r', alpha=0.8)


        ax3.plot(map.ox, map.oy, ".k")
        ax3.scatter(selected_goal[0], selected_goal[1], marker="*", c='black', s=15, label='goal')
        #ax2.imshow(background_image)
        ax3.axis('off')
        ax3.set_aspect('equal', 'box')
        fig3.colorbar(c2, ax=ax2)
        fig3.tight_layout()
        fig3.savefig(f'{savedir_w_name}_ep_goal_fixed_no_back{goal_index}.png',bbox_inches='tight')
        plt.close(fig3)


     ### Another plot
    del aim_output_visualize

    aim_output_visualize = agent.aim_discriminator.forward(inputs_norm_tensor_tmp_initial).detach().cpu().numpy().flatten()
    c = ax2.pcolormesh(pixel_positions[:, 0], pixel_positions[:, 1],aim_output_visualize.reshape(x_debug.shape[0], -1).T,
                      cmap='RdBu_r', alpha=0.8)
    initial_state_pixel = cv2.perspectiveTransform(initial_state.reshape(-1, 1, 2), M).reshape(-1, 2).flatten()
    #Initial Position scatter plot
    ax2.scatter(initial_state_pixel, initial_state_pixel, marker="*", c='black', s=15, label='goal')
    ax2.imshow(background_image)
    ax2.axis('off')
    ax2.set_aspect('equal', 'box')
    fig.colorbar(c, ax=ax)
    fig2.tight_layout()
    fig2.savefig(f'{savedir_w_name}_ep_init_fixed_no_back.png', bbox_inches='tight')

    plt.close(fig2)

    c2 = ax4.pcolormesh(x_debug, y_debug, aim_output_visualize.reshape(x_debug.shape[0], -1).T,cmap='RdBu_r', alpha=0.8)
    ax4.plot(map.ox, map.oy, ".k")
    ax4.scatter(initial_state[0], initial_state[1], marker="*", c='black', s=15, label='goal')
    #ax2.imshow(background_image)
    ax4.axis('off')
    ax4.set_aspect('equal', 'box')
    fig4.colorbar(c2, ax=ax2)
    fig4.tight_layout()
    fig4.savefig(f'{savedir_w_name}_ep_init_fixed.png', bbox_inches='tight')
    plt.close(fig4)

    plt.close('all')




    # for goal_index, selected_goal in enumerate(goal_candidates):
    #     repeated_state_debug = torch.tile(torch.tensor(selected_goal).to(class_instance.device), [virtual_diffusion_goals_debug.shape[0], 1])
    #     inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, repeated_state_debug))
    #     if class_instance.cfg.normalize_rl_obs:
    #         inputs_norm_tensor_tmp = agent.normalize_obs(inputs_norm_tensor_tmp, class_instance.cfg.env)
    #
    #     inputs_norm_tensor_tmp = inputs_norm_tensor_tmp.to(dtype=torch.float32)
    #     aim_output_visualize = agent.aim_discriminator.forward(inputs_norm_tensor_tmp).detach().cpu().numpy().flatten()
    #
    #
    #
    #
    #     M = cv2.getPerspectiveTransform(world_coordinate, image_coordinates)
    #     background_image = mpimg.imread(background_image_path)
    #     positions = np.stack((x_debug, y_debug), axis=1)
    #     pixel_positions = cv2.perspectiveTransform(positions.reshape(-1, 1, 2), M).reshape(-1, 2)
    #     X_debug_pixel, Y_debug_pixel = np.meshgrid(pixel_positions[:, 0], pixel_positions[:, 1])
    #     aim_outputs_tmp_pixel = cv2.perspectiveTransform(aim_output_visualize.reshape(-1, 1, 2), M)
    #     fig, ax = plt.subplots()
    #     C = np.stack((X_debug_pixel, Y_debug_pixel), axis=-1).reshape(-1, 2)
    #     S = np.concatenate([np.zeros((C.shape[0], 7)), C], axis=1)
    #     c = ax.pcolormesh(pixel_positions[:, 0], pixel_positions[:, 1], aim_outputs_tmp_pixel.reshape(x_debug.shape[0], -1),
    #                       cmap='RdBu_r', alpha=0.8)
    #     background_image = mpimg.imread(background_image_path)
    #     # Create the plot and display the background image
    #     ax.imshow(background_image)
    #     ax.axis('off')
    #     ax.axis('tight')
    #     ax.set_aspect('equal', 'box')
    #     fig.colorbar(c)
    #
    #     plt.savefig('/home/erdicitymos/Desktop/DiffusionOutpace/saved_log/PointSpiralMaze-v0/2025.04.04/aim_reward_env.png',bbox_inches='tight')
    #
    #     fig, ax = plt.subplots()
    #     C = np.stack((X_debug, Y_debug), axis=-1).reshape(-1, 2)
    #     S = np.concatenate([np.zeros((C.shape[0], 7)), C], axis=1)
    #     c = ax.pcolormesh(x_debug, y_debug, aim_outputs_tmp_pixel.reshape(x_debug.shape[0], -1), cmap='RdBu_r', alpha=0.8)
    #     map = UMaze()
    #     ax.plot(map.ox, map.oy, ".k")
    #     ax.axis('off')
    #     ax.axis('tight')
    #     ax.set_aspect('equal', 'box')
    #     ax.scatter(0, 8, marker="*", color='red', edgecolor='k', s=65)
    #     ax.scatter(0, 0, marker="*", color='orange', edgecolor='k', s=65)
    #     fig.colorbar(c)




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

