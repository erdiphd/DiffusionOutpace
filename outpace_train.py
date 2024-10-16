import os
from random import random, uniform
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'

import cv2
import copy
import pickle as pkl
import sys

sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette'))
#import infer
from infer import single_img
#from palette import infer
from core.util import tensor2img 
#from inspect import getmembers,isfunction
#print(getmembers(palette,getmembers))

import time
import math
import random

import numpy as np
from queue import Queue
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer, HindsightExperienceReplayWrapperVer2
from video import VideoRecorder
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from hgg.hgg import goal_distance
from visualize.visualize_2d import *
torch.backends.cudnn.benchmark = True

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



class UniformFeasibleGoalSampler:
    def __init__(self, env_name):        
        self.env_name = env_name        
        if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ['sawyer_peg_pick_and_place']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.4])            
        elif env_name in ['sawyer_peg_push']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.02])        
        elif env_name == "PointSpiralMaze-v0":
            self.LOWER_CONTEXT_BOUNDS = np.array([-10, -10]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ["PointNMaze-v0"]:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 18])
        else:
            raise NotImplementedError

    def is_feasible(self, context): 
        # Check that the context is not in or beyond the outer wall
        if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']: # 0.5 margin
            if np.any(context < -1.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            else:
                return True
        elif self.env_name == "PointSpiralMaze-v0":
            if np.any(context < -9.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-6.5 < context[0], context[0] < -1.5), np.logical_and(-6.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 10.5), np.logical_and(-6.5 < context[1], context[1] < -1.5))):
                return False
            else:
                return True
            
        elif self.env_name in ["PointNMaze-v0"]:
            if (context[0] < -1.5) or (context[0] > 9.5):
                return False
            elif (context[1] < -1.5) or (context[1] > 17.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(1.5 < context[0], context[0] < 10.5), np.logical_and(9.5 < context[1], context[1] < 14.5))):
                return False
            else:
                return True
        
        elif self.env_name in ['sawyer_peg_pick_and_place']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        elif self.env_name in ['sawyer_peg_push']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        else:
            raise NotImplementedError

    def sample(self):
        
        sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        while not self.is_feasible(sample):
            sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        
        return sample


def get_object_states_only_from_goal(env_name, goal):
    if env_name in ['sawyer_door', 'sawyer_peg']:
        return goal[..., 4:7]

    elif env_name == 'tabletop_manipulation':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

def get_original_final_goal(env_name):
    if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
        original_final_goal = np.array([0., 8.])
    elif env_name in ['sawyer_peg_push']:
        original_final_goal = np.array([-0.3, 0.4, 0.02])
    elif env_name in ['sawyer_peg_pick_and_place']:
        original_final_goal = np.array([-0.3, 0.4, 0.2])
    elif env_name == "PointSpiralMaze-v0":
        original_final_goal = np.array([8., -8.])
    elif env_name in ["PointNMaze-v0"]:
        original_final_goal = np.array([8., 16.])
    else:
        raise NotImplementedError
    return original_final_goal.copy()


max_episode_timesteps_dict = {'AntMazeSmall-v0' : 300,
                              'PointUMaze-v0' : 100,
                              'sawyer_peg_pick_and_place' : 200,
                              'sawyer_peg_push' : 200,
                              'PointNMaze-v0' : 100, 
                              'PointSpiralMaze-v0' : 200,
                             }

num_seed_steps_dict = { 'AntMazeSmall-v0' : 4000,
                        'PointUMaze-v0' : 2000,
                        'sawyer_peg_pick_and_place' : 2000,
                        'sawyer_peg_push' : 2000,
                        'PointNMaze-v0' : 2000, 
                        'PointSpiralMaze-v0' : 2000,
                        }

num_random_steps_dict = {'AntMazeSmall-v0' : 4000,
                         'PointUMaze-v0' : 2000,
                         'sawyer_peg_pick_and_place' : 2000,
                         'sawyer_peg_push' : 2000,
                         'PointNMaze-v0' : 2000, 
                         'PointSpiralMaze-v0' : 2000,
                        }

randomwalk_random_noise_dict = {'AntMazeSmall-v0' : 2.5,
                                'PointUMaze-v0' : 2.5,
                                'sawyer_peg_pick_and_place' : 0.1,
                                'sawyer_peg_push' : 0.1,
                                'PointNMaze-v0' : 2.5, 
                                'PointSpiralMaze-v0' : 2.5,
                                }
aim_disc_replay_buffer_capacity_dict = {'AntMazeSmall-v0' : 50000,
                                        'PointUMaze-v0' : 10000,
                                        'sawyer_peg_pick_and_place' : 30000,
                                        'sawyer_peg_push' : 30000,
                                        'PointNMaze-v0' : 10000, 
                                        'PointSpiralMaze-v0' : 20000,
                                        }



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent='outpace_rl')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        cfg.max_episode_timesteps = max_episode_timesteps_dict[cfg.env]
        cfg.num_seed_steps = num_seed_steps_dict[cfg.env]
        cfg.num_random_steps = num_random_steps_dict[cfg.env]
        cfg.randomwalk_random_noise = randomwalk_random_noise_dict[cfg.env]
        assert cfg.aim_disc_replay_buffer_capacity == aim_disc_replay_buffer_capacity_dict[cfg.env]
     
        if cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            cfg.goal_env=False
            from envs import sawyer_peg_pick_and_place, sawyer_peg_push
            if cfg.env =='sawyer_peg_pick_and_place':
                env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
                eval_env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
            elif cfg.env =='sawyer_peg_push':
                env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
                eval_env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
            
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_timesteps)
            eval_env = TimeLimit(eval_env, max_episode_steps=cfg.max_episode_timesteps)
            

            if cfg.use_residual_randomwalk:
                from env_utils import ResidualGoalWrapper
                env = ResidualGoalWrapper(env, env_name = cfg.env)
                eval_env = ResidualGoalWrapper(eval_env, env_name = cfg.env)

                                       
            from env_utils import StateWrapper, DoneOnSuccessWrapper
            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                env = DoneOnSuccessWrapper(env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)
                eval_env = DoneOnSuccessWrapper(eval_env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)

            from env_utils import WraptoGoalEnv
            self.env = StateWrapper(WraptoGoalEnv(env, env_name = cfg.env))
            self.eval_env = StateWrapper(WraptoGoalEnv(eval_env, env_name = cfg.env))

            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
        
        elif cfg.goal_env: # e.g. Fetch, Ant
            import gym            
            from env_utils import StateWrapper, HERGoalEnvWrapper, DoneOnSuccessWrapper, ResidualGoalWrapper             
            if cfg.env in ['AntMazeSmall-v0']:
                from gym.wrappers.time_limit import TimeLimit
                from envs.AntEnv.envs.antenv import EnvWithGoal
                from envs.AntEnv.envs.antenv.create_maze_env import create_maze_env                                              
                self.env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                # self.eval_env.evaluate = True # set test goal = (0,16)
                self.env.set_attribute(evaluate=False, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)
                self.eval_env.set_attribute(evaluate=True, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)


                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)
            elif cfg.env in ["PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                from gym.wrappers.time_limit import TimeLimit
                import mujoco_maze                                             
                self.env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)

                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)

            else:
                self.env = gym.make(cfg.env)
                self.eval_env = gym.make(cfg.env)

            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                self.env = DoneOnSuccessWrapper(self.env, relative_goal_env = (relative_goal_env or residual_goal_env))
                self.eval_env = DoneOnSuccessWrapper(self.eval_env, relative_goal_env = (relative_goal_env or residual_goal_env))
            # self.goal_env = self.env
            
            self.env= StateWrapper(HERGoalEnvWrapper(self.env, env_name= cfg.env))
            self.eval_env= StateWrapper(HERGoalEnvWrapper(self.eval_env, env_name= cfg.env))
        
                
            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
            

        
        cfg.agent.action_shape = action_spec.shape
        

        
        cfg.agent.action_range = [
            float(action_spec.low.min()),
            float(action_spec.high.max())
        ]
            
        
            
        self.max_episode_timesteps = cfg.max_episode_timesteps
        
        if cfg.aim_discriminator_cfg.output_activation in [None, 'none', 'None']:
            cfg.aim_discriminator_cfg.output_activation = None
      
            

        if cfg.use_meta_nml:
            if cfg.meta_nml.num_finetuning_layers in [None, 'none', 'None']:
                cfg.meta_nml.num_finetuning_layers = None
            if cfg.meta_nml_kwargs.meta_nml_custom_embedding_key in [None, 'none', 'None']:
                cfg.meta_nml_kwargs.meta_nml_custom_embedding_key = None
            
        cfg.meta_nml.equal_pos_neg_test= cfg.meta_nml_kwargs.equal_pos_neg_test and (not cfg.meta_nml_kwargs.meta_nml_negatives_only)
        cfg.meta_nml.input_dim = self.env.goal_dim
        
        if cfg.env in ['sawyer_door', 'sawyer_peg']:      
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = (get_object_states_only_from_goal(self.cfg.env, np.ones(self.env.goal_dim)).shape[-1])*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        elif cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        else:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            

        
        cfg.agent.goal_dim = self.env.goal_dim

        cfg.agent.obs_shape = obs_spec.shape
        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent)
        
            
        self.expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device)
        
        self.aim_expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.aim_disc_replay_buffer_capacity,
                                        self.device)
        n_sampled_goal = 4
        self.randomwalk_buffer = None
        if cfg.use_residual_randomwalk:
            self.randomwalk_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.randomwalk_buffer_capacity,
                                        self.device)
            self.randomwalk_buffer = HindsightExperienceReplayWrapperVer2(self.randomwalk_buffer, 
                                                                n_sampled_goal=n_sampled_goal, 
                                                                wrapped_env=self.env,
                                                                env_name = cfg.env,
                                                                consider_done_true = cfg.done_on_success,
                                                                )

        self.goal_buffer = None


        
        self.expl_buffer = HindsightExperienceReplayWrapperVer2(self.expl_buffer, 
                                                            n_sampled_goal=n_sampled_goal, 
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )
        self.aim_expl_buffer = HindsightExperienceReplayWrapperVer2(self.aim_expl_buffer, 
                                                            n_sampled_goal=cfg.aim_n_sampled_goal, #n_sampled_goal, 
                                                            # goal_selection_strategy=KEY_TO_GOAL_STRATEGY['future'],
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )
        if cfg.use_hgg:
            from hgg.hgg import TrajectoryPool, MatchSampler            
            self.hgg_achieved_trajectory_pool = TrajectoryPool(**cfg.hgg_kwargs.trajectory_pool_kwargs)
            self.hgg_sampler = MatchSampler(goal_env=self.eval_env, 
                                            goal_eval_env = self.eval_env, 
                                            env_name=cfg.env,
                                            achieved_trajectory_pool = self.hgg_achieved_trajectory_pool,
                                            agent = self.expl_agent,
                                            **cfg.hgg_kwargs.match_sampler_kwargs
                                            )                
       
            
        self.eval_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder.init(enabled=False)
        self.step = 0
        
        
        self.uniform_goal_sampler =  UniformFeasibleGoalSampler(env_name=cfg.env)


    def get_agent(self):                
        return self.expl_agent
        

    def get_buffer(self):                
        return self.expl_buffer
        

    def evaluate(self, eval_uniform_goal=False):
        uniform_goal=False
        repeat = 2 if eval_uniform_goal else 1
                        
        for r in range(repeat):            
            uniform_goal = True if r==1 else False

            avg_episode_reward = 0
            avg_episode_success_rate = 0       
            eval_subgoal_list = []
            for episode in range(self.cfg.num_eval_episodes):
                observes = []
                if uniform_goal:
                    sampled_goal = self.uniform_goal_sampler.sample()                    
                    obs = self.eval_env.reset(goal = sampled_goal)
                else:
                    obs = self.eval_env.reset()            
                
                final_goal = self.eval_env.goal.copy()                    
                        
                observes.append(obs)
                self.eval_video_recorder.init(enabled=False)
                episode_reward = 0
                episode_step = 0
                done = False
                while not done:                
                    agent = self.get_agent()
                    
                    with utils.eval_mode(agent):
                        action = agent.act(obs, spec = self.eval_env.action_spec(), sample=False)
                    next_obs, reward, done, info = self.eval_env.step(action)

                    #if  self.cfg.num_train_steps > self.step:
                    #self.eval_env.render(width=500, height =500)
                    
                    self.eval_video_recorder.record(self.eval_env)
                    episode_reward += reward
                    episode_step += 1
                    obs = next_obs
                    
                    if self.cfg.use_residual_randomwalk:
                        if ((episode_step) % self.max_episode_timesteps == 0) or info.get('is_current_goal_success'):
                            done = True

                    observes.append(obs)
                    

            

                if self.eval_env.is_successful(obs):
                    avg_episode_success_rate+=1.0
                
                if self.cfg.use_aim and episode==0:                
                    fig = plt.figure(figsize=(15,15))
                    sns.set_style("darkgrid")
                    observes = np.stack(observes, axis =0)
                    obs_dict = self.eval_env.convert_obs_to_dict(observes)
                    tiled_initial_obs = np.tile(obs_dict['achieved_goal'][0][None, :], (observes.shape[0], 1)) #[ts, dim]
                    
                    obs_desired_goal = obs_dict['desired_goal']                
                        

                    if self.cfg.env in ['sawyer_door', 'sawyer_peg']:
                        if self.cfg.aim_kwargs.aim_input_type=='default':
                            observes = torch.from_numpy(np.concatenate([get_object_states_only_from_goal(self.cfg.env, obs_dict['achieved_goal']), get_object_states_only_from_goal(self.cfg.env, obs_desired_goal)], axis =-1)).float().to(self.device) #[ts, dim]
                            observes_reverse = torch.from_numpy(np.concatenate([get_object_states_only_from_goal(self.cfg.env, obs_dict['achieved_goal']), get_object_states_only_from_goal(self.cfg.env, tiled_initial_obs)], axis =-1)).float().to(self.device) #[ts, dim]

                    else:
                        if self.cfg.aim_kwargs.aim_input_type=='default':
                            observes = torch.from_numpy(np.concatenate([obs_dict['achieved_goal'], obs_desired_goal], axis =-1)).float().to(self.device) #[ts, dim]
                            observes_reverse = torch.from_numpy(np.concatenate([obs_dict['achieved_goal'], tiled_initial_obs], axis =-1)).float().to(self.device) #[ts, dim]


                    
                    aim_reward = self.expl_agent.compute_aim_reward(observes).detach().cpu().numpy()
                    if self.cfg.normalize_f_obs:
                        aim_disc_outputs = self.expl_agent.aim_discriminator.forward(self.expl_agent.normalize_obs(observes, self.cfg.env)).detach().cpu().numpy()
                    else:
                        aim_disc_outputs = self.expl_agent.aim_discriminator.forward(observes).detach().cpu().numpy()
                    
                    aim_reward_reverse = self.expl_agent.compute_aim_reward(observes_reverse).detach().cpu().numpy()
                    if self.cfg.normalize_f_obs:
                        aim_disc_outputs_reverse = self.expl_agent.aim_discriminator.forward(self.expl_agent.normalize_obs(observes_reverse, self.cfg.env)).detach().cpu().numpy()
                    else:
                        aim_disc_outputs_reverse = self.expl_agent.aim_discriminator.forward(observes_reverse).detach().cpu().numpy()

                    
                    timesteps = np.arange(observes.shape[0])
                    ax1 = fig.add_subplot(4,1,1)                
                    ax1.plot(timesteps, aim_reward, label = 'aim_reward')
                    ax1.legend(loc ='upper right') # , prop={'size': 20}
                    ax2 = fig.add_subplot(4,1,2)
                    ax2.plot(timesteps, aim_disc_outputs, label = 'aim_disc_output')
                    ax2.legend(loc ='upper right')
                    ax3 = fig.add_subplot(4,1,3)                
                    ax3.plot(timesteps, aim_reward_reverse, label = 'aim_reward_reverse')
                    ax3.legend(loc ='upper right') # , prop={'size': 20}
                    ax4 = fig.add_subplot(4,1,4)
                    ax4.plot(timesteps, aim_disc_outputs_reverse, label = 'aim_disc_output_reverse')
                    ax4.legend(loc ='upper right')
                    if uniform_goal:
                        plt.savefig(self.eval_video_recorder.save_dir+'/aim_outputs_uniform_goal_'+str(self.step)+'.jpg')
                    else:
                        plt.savefig(self.eval_video_recorder.save_dir+'/aim_outputs_'+str(self.step)+'.jpg')
                    plt.close()
                    
                avg_episode_reward += episode_reward
                if uniform_goal:
                    self.eval_video_recorder.save(f'uniform_goal_{self.step}.mp4')
                else:
                    self.eval_video_recorder.save(f'{self.step}.mp4')
            avg_episode_reward /= self.cfg.num_eval_episodes
            avg_episode_success_rate = avg_episode_success_rate/self.cfg.num_eval_episodes
            if uniform_goal:                
                self.eval_env.reset(goal = get_original_final_goal(self.cfg.env))
                self.logger.log('eval/episode_reward_uniform_goal', avg_episode_reward, self.step)
                self.logger.log('eval/episode_success_rate_uniform_goal', avg_episode_success_rate, self.step)
            else:
                self.logger.log('eval/episode_reward', avg_episode_reward, self.step)
                self.logger.log('eval/episode_success_rate', avg_episode_success_rate, self.step)

                

        self.logger.dump(self.step, ty='eval')
            
    
    def get_inv_weight_curriculum_buffer(self):        

        if self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='aim':
            return self.aim_expl_buffer
        elif self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='default':
            return self.expl_buffer

    def run(self):        
        self._run()
    
    def _run(self):        
        episode, episode_reward, episode_step = 0, 0, 0
        inv_curriculum_pocket = []
        start_time = time.time()

        if self.cfg.use_hgg:
            recent_sampled_goals = Queue(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes)
        


        previous_goals = None
        done = True
        info = {}

        
        if self.cfg.use_meta_nml:                
            if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:  
                final_goal_states = np.tile(np.array([0., 8.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env  == "PointSpiralMaze-v0":
                final_goal_states = np.tile(np.array([8., -8.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env in ["PointNMaze-v0"]:
                final_goal_states = np.tile(np.array([8., 16.]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
            elif self.cfg.env in [    'sawyer_peg_push' ]:
                final_goal_states = np.tile(np.array([-0.3, 0.4, 0.02]), (self.cfg.aim_num_precollect_init_state,1))
                noise = np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.05*np.ones_like(final_goal_states))
                noise[2] = 0
                final_goal_states += noise
            elif self.cfg.env in ['sawyer_peg_pick_and_place']:
                final_goal_states = np.tile(np.array([-0.3, 0.4, 0.2]), (self.cfg.aim_num_precollect_init_state,1))
                final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.05*np.ones_like(final_goal_states))
            else:
                raise NotImplementedError
            agent = self.get_agent()
            agent.final_goal_states = final_goal_states.copy()

        if self.cfg.use_hgg:
            temp_obs = self.eval_env.reset()        
            recent_sampled_goals.put(self.eval_env.convert_obs_to_dict(temp_obs)['achieved_goal'].copy())


        current_pocket_success = 0
        current_pocket_trial = 0

        positions = []

        action_list = []
        observe_list = []


        while self.step <= self.cfg.num_train_steps:
            
            if done:
                if self.step > 0:
                    current_pocket_trial += 1
                    if info['is_success']:
                        current_pocket_success += 1 
                    
                    # hgg update
                    if self.cfg.use_hgg :
                        if episode % self.cfg.hgg_kwargs.hgg_sampler_update_frequency ==0 :                            
                            initial_goals = []
                            desired_goals = []
                            # collect s_0, g from T*                            
                            for i in range(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes):                                
                                temp_obs = self.eval_env.convert_obs_to_dict(self.eval_env.reset())
                                goal_a = temp_obs['achieved_goal'].copy()                                
                                if 'meta_nml' in hgg_sampler.cost_type or 'aim_f' in hgg_sampler.cost_type: 
                                    # In this case, desired_goal is not used inside
                                    # for preventing initial sampled hgg goals to be final goal
                                    if self.cfg.env in ['AntMazeSmall-v0', 'PointUMaze-v0', 'PointNMaze-v0', 'PointSpiralMaze-v0']:
                                        noise_scale = 0.5                    
                                        noise = np.random.normal(loc=np.zeros_like(goal_a), scale=noise_scale*np.ones_like(goal_a))
                                    elif self.cfg.env in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
                                        noise_scale = 0.05
                                        noise = np.random.normal(loc=np.zeros_like(goal_a), scale=noise_scale*np.ones_like(goal_a))
                                        noise[2] = 0 # zero out z element to prevent sampling through the table
                                    else:
                                        raise NotImplementedError
                                    goal_d = goal_a + noise # These will be meaningless after achieved_goals are accumulated in hgg_achieved_trajectory_pool
                                else:
                                    raise NotImplementedError
                                initial_goals.append(goal_a.copy())
                                desired_goals.append(goal_d.copy()) 
                            hgg_start_time = time.time()
                            hgg_sampler = self.hgg_sampler
                            hgg_sampler.update(initial_goals, desired_goals, replay_buffer = self.expl_buffer, meta_nml_epoch=episode) 
                            # print('hgg sampler update step : {} time : {}'.format(self.step, time.time() - hgg_start_time))
                    




                self.train_video_recorder.save(f'train_episode_{episode-1}.mp4')                
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                
                if self.cfg.use_hgg:                    
                    hgg_sampler = self.hgg_sampler
                    n_iter = 0
                    while True:
                        # print('hgg sampler pool len : {} step : {}'.format(len(hgg_sampler.pool), self.step))
                        sampled_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool))).copy()                        #THIS LINE GIVES CURRICULUM GOAL
                        obs = self.env.reset(goal = sampled_goal)

                        if not self.env.is_successful(obs):
                            break
                        n_iter +=1
                        if n_iter==10:
                            break
                    if recent_sampled_goals.full():
                        recent_sampled_goals.get()
                    recent_sampled_goals.put(sampled_goal)
                    # obs = self.env.reset(goal = sampled_goal)
                    assert (sampled_goal == self.env.goal.copy()).all()
                else:
                    agent = self.get_agent()
                    if self.step != 0 and self.cfg.use_inpainting:
                        obs = self.env.reset(goal = inpaint_goal) # this is state_0 # u maze de 10 boyutlu
                        #generated_curriculum_goal.append(obs)
                    else: 
                        obs = self.env.reset()
                final_goal = self.env.goal.copy()            
                print(f"Episode:{episode}, final_goal = {final_goal}")
                    
                self.logger.log('train/episode_finalgoal_dist', np.linalg.norm(final_goal), self.step)
                if self.cfg.use_hgg:
                    original_final_goal = get_original_final_goal(self.cfg.env)                    
                    self.logger.log('train/episode_dist_from_curr_g_to_example_g', np.linalg.norm(final_goal-original_final_goal), self.step)                                        
                    sampled_goals_for_log = np.array(recent_sampled_goals.queue)
                    self.logger.log('train/average_dist_from_curr_g_to_example_g', np.linalg.norm(original_final_goal[None, :]-sampled_goals_for_log, axis =-1).mean(), self.step)
                        


                self.train_video_recorder.init(enabled=False)
                
                hgg_save_freq = 3 if 'Point' in self.cfg.env else 25
                if self.cfg.use_hgg and episode % hgg_save_freq == 0 :
                    sampled_goals_for_vis = np.array(recent_sampled_goals.queue) 
                    fig = plt.figure()
                    sns.set_style("darkgrid")
                    ax1 = fig.add_subplot(1,1,1)                    
                    ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1])
                    if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
                        plt.xlim(-2,10)    
                        plt.ylim(-2,10)
                    elif self.cfg.env == "PointSpiralMaze-v0":
                        plt.xlim(-10,10)    
                        plt.ylim(-10,10)
                    elif self.cfg.env in ["PointNMaze-v0"]:
                        plt.xlim(-2,10)    
                        plt.ylim(-2,18)
                    elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                        plt.xlim(-0.6,0.6)    
                        plt.ylim(0.2,1.0)
                    else:
                        raise NotImplementedError
                    plt.savefig(self.train_video_recorder.save_dir+'/train_hgg_goals_episode_'+str(episode)+'.jpg')
                    plt.close()
                    with open(self.train_video_recorder.save_dir+'/train_hgg_goals_episode_'+str(episode)+'.pkl', 'wb') as f:
                        pkl.dump(sampled_goals_for_vis, f)


                if episode % self.cfg.train_episode_video_freq == 0 or episode in [25,50,75,100]:             
                    self.train_video_recorder.init(enabled=False)           
                    # Visualize from init state to subgoals
                   

                    visualize_num_iter = 0
                    scatter_states = self.env.convert_obs_to_dict(obs.copy())['achieved_goal'][None, :]
                    
                    for k in range(visualize_num_iter+1):                                                
                        init_state = scatter_states[k]
                        if self.cfg.use_aim:
                            visualize_discriminator(normalizer = agent.normalize_obs if self.cfg.normalize_f_obs else None,
                                                    discriminator = agent.aim_discriminator, 
                                                    initial_state = init_state, 
                                                    scatter_states = scatter_states.squeeze(),
                                                    env_name = self.cfg.env, 
                                                    aim_input_type = self.cfg.aim_kwargs.aim_input_type,
                                                    device = self.device, 
                                                    savedir_w_name = self.train_video_recorder.save_dir + '/aim_f_visualize_train_episode_'+str(episode)+'_s'+str(k),
                                                    )

                            visualize_discriminator2(normalizer = agent.normalize_obs if self.cfg.normalize_f_obs else None,
                                                    discriminator = agent.aim_discriminator, 
                                                    env_name = self.cfg.env, 
                                                    aim_input_type = self.cfg.aim_kwargs.aim_input_type,
                                                    device = self.device, 
                                                    savedir_w_name = self.train_video_recorder.save_dir + '/aim_f_visualize_train_goalfix_'+str(episode)+'_s'+str(k),
                                                    )
                        if self.cfg.use_meta_nml:
                            visualize_meta_nml(agent=agent, 
                                               meta_nml_epoch=episode, 
                                               scatter_states = scatter_states.squeeze(),
                                               replay_buffer= self.get_buffer(), 
                                               goal_env = self.env,
                                               env_name = self.cfg.env, 
                                               aim_input_type = self.cfg.aim_kwargs.aim_input_type, 
                                               savedir_w_name = self.train_video_recorder.save_dir + '/aim_meta_nml_prob_visualize_train_episode_'+str(episode)+'_s'+str(k),
                                               )
#                if episode == 0:
#                    next_goal = self.env.goal.copy()
#               else:
                if self.cfg.render_images:
                    #print("EPISODE: ", episode)
                    #print("Goal:", self.env.goal)
                    next_goal = self.env.goal.copy()
                    positions = np.array(positions)
                    #print(positions)
                    
                    if self.cfg.env == "PointSpiralMaze-v0":
                        image_scale = 25
                        agent_size = round(image_scale*1.5) #updated for spiral maze to be changed
                    elif self.cfg.env == "sawyer_peg_push":
                        image_scale = 250
                        agent_size = image_scale//15
                    elif self.cfg.env == "sawyer_peg_pick_and_place":
                        image_scale = 250
                        agent_size = image_scale//15
                        z_ratio = 4 #Since the max dimensions of x and z are different by almost 3 times and z should be more precise, I am scaling z by 4
                    else:
                        image_scale = 25
                        agent_size = image_scale

                    goal_size = agent_size//2
                    maze_dim_y = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]
                    maze_dim_x = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]

                    #if self.cfg.env == "sawyer_peg_push":
                    #    maze_dim_x = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]
                    #    maze_dim_y = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]

                    cur_image_bw = np.zeros((int(maze_dim_x*image_scale),int(maze_dim_y*image_scale)))

                    if self.cfg.env == "sawyer_peg_pick_and_place":
                        maze_dim_z = (self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[2]-self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2])
                        z_scale = int(image_scale*z_ratio)
                        cur_image_bw_yz = np.zeros((int(maze_dim_y*image_scale),int(maze_dim_z*z_scale)))                        

                    prev_val_y = 0
                    prev_val_x = 0

                    for i in range(len(positions)):
                        if self.cfg.env == "sawyer_peg_pick_and_place":
                            agent_size = image_scale//15
                            #z_ratio = 4 #Since the max dimensions of x and z are different by almost 3 times and z should be more precise, I am scaling z by 4
                        cur_val_y = int(round((positions[i][0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))
                        cur_val_x = int((maze_dim_x*image_scale)-1 - round((positions[i][1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1])*image_scale))

                        if i == 0:
                            prev_val_y = cur_val_y
                            prev_val_x = cur_val_x

                        if cur_val_x+(agent_size//2 + 1) >= maze_dim_x*image_scale:
                            #cur_val_x -= (agent_size//2)+1
                            cur_val_x = int((maze_dim_x*image_scale) - ((agent_size//2) +2))
                        if cur_val_y+(agent_size//2 + 1) >= maze_dim_y*image_scale:
                            cur_val_y = int((maze_dim_y*image_scale) - ((agent_size//2) +2))
                            #cur_val_y -= (agent_size//2)+1

                        if cur_val_x-(agent_size//2 + 1) <= 0:
                            #cur_val_x += (agent_size//2)+1
                            cur_val_x = int((agent_size//2)+2)
                        if cur_val_y-(agent_size//2 + 1) <= 0:
                            #cur_val_y += (agent_size//2)+1
                            cur_val_y = int((agent_size//2)+2)

                        max_movement = math.ceil(math.sqrt(abs(cur_val_y-prev_val_y)*abs(cur_val_y-prev_val_y) + abs(prev_val_x-cur_val_x)*abs(prev_val_x-cur_val_x)))

                        if (self.cfg.env == "sawyer_peg_push" or self.cfg.env == "sawyer_peg_pick_and_place") and max_movement == 0 and i == 0: #to show if the object does not move since max move is 0 does not go down
                            for k in range(agent_size):
                                for n in range(agent_size):
                                    if ((k < agent_size//3 and n < agent_size//3) or (k > (agent_size-agent_size//3) and n > (agent_size-agent_size//3)) 
                                        or (k < agent_size//3 and n > (agent_size-agent_size//3)) or (k > (agent_size-agent_size//3) and n < agent_size//3)):
                                        continue
                                    cur_image_bw[prev_val_x-(agent_size//2)+k][prev_val_y-(agent_size//2)+n] += 1

                        current_coordinate_y = prev_val_y
                        current_coordinate_x = prev_val_x
                        for j in range(max_movement): #move coordinate towards next coordinate and mark the ones along the way

                            if abs(current_coordinate_y - cur_val_y) > 0 and abs(current_coordinate_x - cur_val_x) > 0: #move diagonally
                                if current_coordinate_y > cur_val_y and current_coordinate_x > cur_val_x: #left bottom
                                    current_coordinate_y -= 1
                                    current_coordinate_x -= 1
                                elif current_coordinate_y < cur_val_y and current_coordinate_x < cur_val_x: #right up
                                    current_coordinate_y += 1
                                    current_coordinate_x += 1
                                elif current_coordinate_y > cur_val_y and current_coordinate_x < cur_val_x: #left up
                                    current_coordinate_y -= 1
                                    current_coordinate_x += 1
                                elif current_coordinate_y < cur_val_y and current_coordinate_x > cur_val_x: #right down
                                    current_coordinate_y += 1
                                    current_coordinate_x -= 1
                                
                            elif abs(current_coordinate_y - cur_val_y) > 0 and abs(current_coordinate_x - cur_val_x) == 0: # move in y
                                if current_coordinate_y < cur_val_y:
                                    current_coordinate_y += 1
                                elif current_coordinate_y > cur_val_y:
                                    current_coordinate_y -= 1
                            elif abs(current_coordinate_y - cur_val_y) == 0 and abs(current_coordinate_x - cur_val_x) > 0: # move in x
                                if current_coordinate_x < cur_val_x:
                                    current_coordinate_x += 1
                                elif current_coordinate_x > cur_val_x:
                                    current_coordinate_x -= 1
                            elif abs(current_coordinate_y - cur_val_y) == 0 and abs(current_coordinate_x - cur_val_x) == 0:
                                break

                            for k in range(agent_size):
                                for n in range(agent_size):
                                    if ((k < agent_size//3 and n < agent_size//3) or (k > (agent_size-agent_size//3) and n > (agent_size-agent_size//3)) 
                                        or (k < agent_size//3 and n > (agent_size-agent_size//3)) or (k > (agent_size-agent_size//3) and n < agent_size//3)):
                                        continue
                                    cur_image_bw[current_coordinate_x-(agent_size//2)+k][current_coordinate_y-(agent_size//2)+n] += 1
                            #cur_image_bw[current_coordinate_x][current_coordinate_y] += 1
                        #cur_image_bw[cur_val_x][cur_val_y] += 1
                        prev_val_y = cur_val_y
                        prev_val_x = cur_val_x

                    if np.max(cur_image_bw) == 0:
                        cur_image_bw = (cur_image_bw-np.min(cur_image_bw))/(1-np.min(cur_image_bw))
                    else:
                        cur_image_bw = (cur_image_bw-np.min(cur_image_bw))/(np.max(cur_image_bw)-np.min(cur_image_bw))

                    cur_image = np.zeros([int(maze_dim_x*image_scale),int(maze_dim_y*image_scale),3])
                    #print(cur_image.shape)
                    #print(cur_image_bw.shape)
                    for i in range(len(cur_image)):
                        for j in range(len(cur_image[i])):
                            cur_image[i][j][0] = 1-cur_image_bw[i][j]
                            cur_image[i][j][1] = 1-cur_image_bw[i][j]
                            cur_image[i][j][2] = 1
                    
                    cur_image_black = copy.deepcopy(cur_image)
                    cur_image_large_goal = copy.deepcopy(cur_image)
                    cur_image_large_goal_black = copy.deepcopy(cur_image)
                    cur_image_no_goal = copy.deepcopy(cur_image)
                    cur_image_largest_goal_black = copy.deepcopy(cur_image)

                    #default goal
                    
                    cur_goal_y = int(round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))
                    cur_goal_x = int((maze_dim_x*image_scale)-1 - round((next_goal[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1])*image_scale))

                    #if self.cfg.env == "sawyer_peg_push":
                    #    cur_goal_y = int(int(round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale)))
                    #    cur_goal_x = int(int((maze_dim_x*image_scale)-1 - round((next_goal[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1])*image_scale)))

                    #print(f"goal_size {goal_size}, goal size//2 {goal_size//2}", goal_size)
                    #print("Maze dims", maze_dim_x,maze_dim_y)
                    #print("Cur goal is", cur_goal_x,cur_goal_y)
                    #print("cur_image_black shape", cur_image.shape)

                    if cur_goal_x +(goal_size//2 + 1) >= maze_dim_x*image_scale:
                        #cur_goal_x -= (goal_size//2)+1
                        cur_goal_x = int((maze_dim_x*image_scale) - ((goal_size//2) +2))
                    if cur_goal_y +goal_size//2 >=maze_dim_y*image_scale:
                        #cur_goal_y -= (goal_size//2)+1
                        cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                    if cur_goal_x-(goal_size//2 + 1) <= 0:
                        #cur_goal_x += (goal_size//2)+1
                        cur_goal_x = int((goal_size//2)+2)
                    if cur_goal_y-(goal_size//2 + 1) <= 0:
                        #cur_goal_y += (goal_size//2)+1
                        cur_goal_y = int((goal_size//2)+2)

                    for k in range(goal_size):
                        for n in range(goal_size):
                            cur_image[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [1,0.8,0.2]
                            cur_image_black[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [0,0,0]
                            #cur_image[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]

                    if not os.path.exists("heatmaps"):
                        os.makedirs("heatmaps")
                    cv2.imwrite(f"heatmaps/heatmap_episode{episode}.jpg", (cur_image*255).astype('uint8'))

                    if not os.path.exists("heatmaps_black"):
                        os.makedirs("heatmaps_black")
                    cv2.imwrite(f"heatmaps_black/heatmap_episode{episode}.jpg", (cur_image_black*255).astype('uint8'))

                    #Larger Goal
                    goal_size = agent_size

                    cur_goal_y = int(round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))
                    cur_goal_x = int((maze_dim_x*image_scale)-1 - round((next_goal[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1])*image_scale))

                    if cur_goal_x +(goal_size//2 + 1) >= maze_dim_x*image_scale:
                        #cur_goal_x -= (goal_size//2)+1
                        cur_goal_x = int((maze_dim_x*image_scale) - ((goal_size//2) +2))
                    if cur_goal_y +goal_size//2 >=maze_dim_y*image_scale:
                        #cur_goal_y -= (goal_size//2)+1
                        cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                    if cur_goal_x-(goal_size//2 + 1) <= 0:
                        #cur_goal_x += (goal_size//2)+1
                        cur_goal_x = int((goal_size//2)+2)
                    if cur_goal_y-(goal_size//2 + 1) <= 0:
                        #cur_goal_y += (goal_size//2)+1
                        cur_goal_y = int((goal_size//2)+2)

                    for k in range(goal_size):
                        for n in range(goal_size):
                            cur_image_large_goal[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [1,0.8,0.2]
                            cur_image_large_goal_black[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [0,0,0]
                            #cur_image[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]
                    
                    #cv2.imshow('Color image', cur_image_large_goal)
                    #cv2.waitKey(5000)
                    if not os.path.exists("heatmaps_large"):
                        os.makedirs("heatmaps_large")
                    cv2.imwrite(f"heatmaps_large/heatmap_episode{episode}.jpg", (cur_image_large_goal*255).astype('uint8'))

                    #cv2.imshow('Color image', cur_image_large_goal_black)
                    #cv2.waitKey(500)
                    if not os.path.exists("heatmaps_large_black"):
                        os.makedirs("heatmaps_large_black")
                    cv2.imwrite(f"heatmaps_large_black/heatmap_episode{episode}.jpg", (cur_image_large_goal_black*255).astype('uint8'))

                    #Even Larger Goal
                    goal_size = agent_size*2

                    cur_goal_y = int(round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))
                    cur_goal_x = int((maze_dim_x*image_scale)-1 - round((next_goal[1] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1])*image_scale))

                    if cur_goal_x +(goal_size//2 + 1) >= maze_dim_x*image_scale:
                        #cur_goal_x -= (goal_size//2)+1
                        cur_goal_x = int((maze_dim_x*image_scale) - ((goal_size//2) +2))
                    if cur_goal_y +goal_size//2 >=maze_dim_y*image_scale:
                        #cur_goal_y -= (goal_size//2)+1
                        cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                    if cur_goal_x-(goal_size//2 + 1) <= 0:
                        #cur_goal_x += (goal_size//2)+1
                        cur_goal_x = int((goal_size//2)+2)
                    if cur_goal_y-(goal_size//2 + 1) <= 0:
                        #cur_goal_y += (goal_size//2)+1
                        cur_goal_y = int((goal_size//2)+2)

                    for k in range(goal_size):
                        for n in range(goal_size):
                            cur_image_largest_goal_black[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [0,0,0]
                            #cur_image[cur_goal_x-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]
                    
                    #cv2.imshow('Color image', cur_image_no_goal)
                    #cv2.waitKey(5000)
                    if not os.path.exists("heatmaps_no_goal"):
                        os.makedirs("heatmaps_no_goal")
                    cv2.imwrite(f"heatmaps_no_goal/heatmap_episode{episode}.jpg", (cur_image_no_goal*255).astype('uint8'))

                    #cv2.imshow('Color image', cur_image_largest_goal_black)
                    #cv2.waitKey(500)
                    if not os.path.exists("heatmaps_largest_black"):
                        os.makedirs("heatmaps_largest_black")
                    cv2.imwrite(f"heatmaps_largest_black/heatmap_episode{episode}.jpg", (cur_image_largest_goal_black*255).astype('uint8'))

                    #next_goal = self.env.goal.copy() if at the end saves with current goal else saves with the next goal


                    if self.cfg.env in ['sawyer_peg_pick_and_place']:
                        
                        agent_size = image_scale//10
                        
                        for i in range(len(positions)):
                            cur_val_z = int(round((positions[i][2] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2])*z_scale))
                            cur_val_y = int((maze_dim_y*image_scale)-1 - round((positions[i][0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))

                            if i == 0:
                                prev_val_z = cur_val_z
                                prev_val_y = cur_val_y

                            if cur_val_y+(agent_size//2 + 1) >= maze_dim_y*image_scale:
                                #cur_val_y -= (agent_size//2)+1
                                cur_val_y = int((maze_dim_y*image_scale) - ((agent_size//2) +2))
                            if cur_val_z+(agent_size//2 + 1) >= maze_dim_z*z_scale:
                                cur_val_z = int((maze_dim_z*z_scale) - ((agent_size//2) +2))
                                #cur_val_z -= (agent_size//2)+1

                            if cur_val_y-(agent_size//2 + 1) <= 0:
                                #cur_val_y += (agent_size//2)+1
                                cur_val_y = int((agent_size//2)+2)
                            if cur_val_z-(agent_size//2 + 1) <= 0:
                                #cur_val_z += (agent_size//2)+1
                                cur_val_z = int((agent_size//2)+2)

                            max_movement = math.ceil(math.sqrt(abs(cur_val_z-prev_val_z)*abs(cur_val_z-prev_val_z) + abs(prev_val_y-cur_val_y)*abs(prev_val_y-cur_val_y)))

                            if max_movement == 0 and i == 0: #to show if the object does not move since max move is 0 does not go down
                                for k in range(agent_size):
                                    for n in range(agent_size):
                                        if ((k < agent_size//3 and n < agent_size//3) or (k > (agent_size-agent_size//3) and n > (agent_size-agent_size//3)) 
                                            or (k < agent_size//3 and n > (agent_size-agent_size//3)) or (k > (agent_size-agent_size//3) and n < agent_size//3)):
                                            continue
                                        cur_image_bw_yz[prev_val_y-(agent_size//2)+k][prev_val_z-(agent_size//2)+n] += 1

                            current_coordinate_z = prev_val_z
                            current_coordinate_y = prev_val_y
                            for j in range(max_movement): #move coordinate towards next coordinate and mark the ones along the way

                                if abs(current_coordinate_z - cur_val_z) > 0 and abs(current_coordinate_y - cur_val_y) > 0: #move diagonally
                                    if current_coordinate_z > cur_val_z and current_coordinate_y > cur_val_y: #left bottom
                                        current_coordinate_z -= 1
                                        current_coordinate_y -= 1
                                    elif current_coordinate_z < cur_val_z and current_coordinate_y < cur_val_y: #right up
                                        current_coordinate_z += 1
                                        current_coordinate_y += 1
                                    elif current_coordinate_z > cur_val_z and current_coordinate_y < cur_val_y: #left up
                                        current_coordinate_z -= 1
                                        current_coordinate_y += 1
                                    elif current_coordinate_z < cur_val_z and current_coordinate_y > cur_val_y: #right down
                                        current_coordinate_z += 1
                                        current_coordinate_y -= 1
                                    
                                elif abs(current_coordinate_z - cur_val_z) > 0 and abs(current_coordinate_y - cur_val_y) == 0: # move in y
                                    if current_coordinate_z < cur_val_z:
                                        current_coordinate_z += 1
                                    elif current_coordinate_z > cur_val_z:
                                        current_coordinate_z -= 1
                                elif abs(current_coordinate_z - cur_val_z) == 0 and abs(current_coordinate_y - cur_val_y) > 0: # move in x
                                    if current_coordinate_y < cur_val_y:
                                        current_coordinate_y += 1
                                    elif current_coordinate_y > cur_val_y:
                                        current_coordinate_y -= 1
                                elif abs(current_coordinate_z - cur_val_z) == 0 and abs(current_coordinate_y - cur_val_y) == 0:
                                    break

                                for k in range(agent_size):
                                    for n in range(agent_size):
                                        if ((k < agent_size//3 and n < agent_size//3) or (k > (agent_size-agent_size//3) and n > (agent_size-agent_size//3)) 
                                            or (k < agent_size//3 and n > (agent_size-agent_size//3)) or (k > (agent_size-agent_size//3) and n < agent_size//3)):
                                            continue
                                        cur_image_bw_yz[current_coordinate_y-(agent_size//2)+k][current_coordinate_z-(agent_size//2)+n] += 1
                                #cur_image_bw_yz[current_coordinate_y][current_coordinate_z] += 1
                            #cur_image_bw_yz[cur_val_y][cur_val_z] += 1
                            prev_val_z = cur_val_z
                            prev_val_y = cur_val_y

                        if np.max(cur_image_bw_yz) == 0:
                            cur_image_bw_yz = (cur_image_bw_yz-np.min(cur_image_bw_yz))/(1-np.min(cur_image_bw_yz))
                        else:
                            cur_image_bw_yz = (cur_image_bw_yz-np.min(cur_image_bw_yz))/(np.max(cur_image_bw_yz)-np.min(cur_image_bw_yz))

                        cur_image_yz = np.zeros([int(maze_dim_y*image_scale),int(maze_dim_z*z_scale),3])
                        #print(cur_image.shape)
                        #print(cur_image_bw_yz.shape)
                        for i in range(len(cur_image_yz)):
                            for j in range(len(cur_image_yz[i])):
                                cur_image_yz[i][j][0] = 1-cur_image_bw_yz[i][j]
                                cur_image_yz[i][j][1] = 1-cur_image_bw_yz[i][j]
                                cur_image_yz[i][j][2] = 1
                        
                        cur_image_yz_black = copy.deepcopy(cur_image_yz)
                        cur_image_yz_large_goal = copy.deepcopy(cur_image_yz)
                        cur_image_yz_large_goal_black = copy.deepcopy(cur_image_yz)
                        cur_image_yz_no_goal = copy.deepcopy(cur_image_yz)
                        cur_image_yz_largest_goal_black = copy.deepcopy(cur_image_yz)

                        #default goal
                        goal_size = agent_size//2
                        
                        cur_goal_z = int(round((next_goal[2] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2])*z_scale))
                        cur_goal_y = int((maze_dim_y*image_scale)-1 - round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))

                        #if self.cfg.env == "sawyer_peg_push":
                        #    cur_goal_y = int(int(round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale)))
                        #    cur_goal_y = int(int((maze_dim_y*image_scale)-1 - round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale)))

                        #print(f"goal_size {goal_size}, goal size//2 {goal_size//2}", goal_size)
                        #print("Maze dims", maze_dim_y,maze_dim_z)
                        #print("Cur goal is", cur_goal_y,cur_goal_y)
                        #print("cur_image_yz_black shape", cur_image.shape)

                        if cur_goal_y +(goal_size//2 + 1) >= maze_dim_y*image_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                        if cur_goal_z +(goal_size//2 + 1) >=maze_dim_z*z_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_z = int((maze_dim_z*z_scale) - ((goal_size//2) +2))
                        if cur_goal_y-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_y = int((goal_size//2)+2)
                        if cur_goal_z-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_z = int((goal_size//2)+2)

                        for k in range(goal_size):
                            for n in range(goal_size):
                                cur_image_yz[cur_goal_y-(goal_size//2)+k][cur_goal_z-(goal_size//2)+n] = [1,0.8,0.2]
                                cur_image_yz_black[cur_goal_y-(goal_size//2)+k][cur_goal_z-(goal_size//2)+n] = [0,0,0]
                                #cur_image_yz[cur_goal_y-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]

                        if not os.path.exists("heatmaps_yz"):
                            os.makedirs("heatmaps_yz")
                        cv2.imwrite(f"heatmaps_yz/heatmap_yz_episode{episode}.jpg", (cur_image_yz*255).astype('uint8'))

                        if not os.path.exists("heatmaps_yz_black"):
                            os.makedirs("heatmaps_yz_black")
                        cv2.imwrite(f"heatmaps_yz_black/heatmap_episode{episode}.jpg", (cur_image_yz_black*255).astype('uint8'))

                        #Larger Goal
                        goal_size = agent_size

                        cur_goal_z = int(round((next_goal[2] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2])*z_scale))
                        cur_goal_y = int((maze_dim_y*image_scale)-1 - round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))

                        if cur_goal_y +(goal_size//2 + 1) >= maze_dim_y*image_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                        if cur_goal_z +(goal_size//2 + 1) >=maze_dim_z*z_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_z = int((maze_dim_z*z_scale) - ((goal_size//2) +2))
                        if cur_goal_y-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_y = int((goal_size//2)+2)
                        if cur_goal_z-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_z = int((goal_size//2)+2)

                        for k in range(goal_size):
                            for n in range(goal_size):
                                cur_image_yz_large_goal[cur_goal_y-(goal_size//2)+k][cur_goal_z-(goal_size//2)+n] = [1,0.8,0.2]
                                cur_image_yz_large_goal_black[cur_goal_y-(goal_size//2)+k][cur_goal_z-(goal_size//2)+n] = [0,0,0]
                                #cur_image_yz[cur_goal_y-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]
                        
                        #cv2.imshow('Color image', cur_image_yz_large_goal)
                        #cv2.waitKey(5000)
                        if not os.path.exists("heatmaps_yz_large"):
                            os.makedirs("heatmaps_yz_large")
                        cv2.imwrite(f"heatmaps_yz_large/heatmap_episode{episode}.jpg", (cur_image_yz_large_goal*255).astype('uint8'))

                        #cv2.imshow('Color image', cur_image_yz_large_goal_black)
                        #cv2.waitKey(500)
                        if not os.path.exists("heatmaps_yz_large_black"):
                            os.makedirs("heatmaps_yz_large_black")
                        cv2.imwrite(f"heatmaps_yz_large_black/heatmap_episode{episode}.jpg", (cur_image_yz_large_goal_black*255).astype('uint8'))

                        #Even Larger Goal
                        goal_size = agent_size*2

                        cur_goal_z = int(round((next_goal[2] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2])*z_scale))
                        cur_goal_y = int((maze_dim_y*image_scale)-1 - round((next_goal[0] - self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0])*image_scale))

                        if cur_goal_y +(goal_size//2 + 1) >= maze_dim_y*image_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_y = int((maze_dim_y*image_scale) - ((goal_size//2) +2))
                        if cur_goal_z +(goal_size//2 + 1) >=maze_dim_z*z_scale:
                            #cur_goal_y -= (goal_size//2)+1
                            cur_goal_z = int((maze_dim_z*z_scale) - ((goal_size//2) +2))
                        if cur_goal_y-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_y = int((goal_size//2)+2)
                        if cur_goal_z-(goal_size//2 + 1) <= 0:
                            #cur_goal_y += (goal_size//2)+1
                            cur_goal_z = int((goal_size//2)+2)
                        
                        for k in range(goal_size):
                            for n in range(goal_size):
                                cur_image_yz_largest_goal_black[cur_goal_y-(goal_size//2)+k][cur_goal_z-(goal_size//2)+n] = [0,0,0]
                                #cur_image_yz[cur_goal_y-(goal_size//2)+k][cur_goal_y-(goal_size//2)+n] = [random.randrange(90,100)/100,random.randrange(75,85)/100,random.randrange(15,25)/100]
                        
                        #cv2.imshow('Color image', cur_image_yz_no_goal)
                        #cv2.waitKey(5000)
                        if not os.path.exists("heatmaps_yz_no_goal"):
                            os.makedirs("heatmaps_yz_no_goal")
                        cv2.imwrite(f"heatmaps_yz_no_goal/heatmap_episode{episode}.jpg", (cur_image_yz_no_goal*255).astype('uint8'))

                        #cv2.imshow('Color image', cur_image_yz_largest_goal_black)
                        #cv2.waitKey(500)
                        if not os.path.exists("heatmaps_yz_largest_black"):
                            os.makedirs("heatmaps_yz_largest_black")
                        cv2.imwrite(f"heatmaps_yz_largest_black/heatmap_episode{episode}.jpg", (cur_image_yz_largest_goal_black*255).astype('uint8'))

                        #next_goal = self.env.goal.copy() if at the end saves with current goal else saves with the next goal


                
                if self.cfg.use_inpainting:
                    cur_model_dir = ""
                    if self.cfg.env in ['AntMazeSmall-v0', 'PointUMaze-v0']:
                        cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_u_maze_64_path_oriented/checkpoint/300_Network.pth')
                    elif self.cfg.env in ['sawyer_peg_pick_and_place']:
                        cur_model_dir_xy = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_sawyer_pick_place_xy_64_path_oriented/checkpoint/300_Network.pth')
                        cur_model_dir_yz = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_sawyer_pick_place_yz_64_path_oriented/checkpoint/300_Network.pth')
                    elif self.cfg.env in ['sawyer_peg_push']:
                        cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_sawyer_push_path_oriented/checkpoint/300_Network.pth')       
                    elif self.cfg.env == "PointSpiralMaze-v0":
                        cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_spiral_maze_64_path_oriented/checkpoint/300_Network.pth')       
                    elif self.cfg.env in ["PointNMaze-v0"]:
                        cur_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'palette/experiments/inpainting_n_maze_64_path_oriented/checkpoint/300_Network.pth')       
                        
                    if not os.path.exists(f"inpaint_results/episode_{episode}") and self.cfg.env not in ['sawyer_peg_pick_and_place'] :
                        os.makedirs(f"inpaint_results/episode_{episode}")

                    IMG_DIMS = 64
                    goal_coordinates = []
                    selected_images = []

                    if self.cfg.env in ['sawyer_peg_pick_and_place']:
                        selected_images_xy = []
                        selected_images_yz = []
                    while not goal_coordinates:
                        if self.cfg.env not in ['sawyer_peg_pick_and_place']:
                            output_imgs = single_img((cur_image_no_goal*255).astype('uint8'),True, f"inpaint_results/episode_{episode}/",model_pth = cur_model_dir,beta_schedule_n_steps = self.cfg.beta_schedule_n_steps)
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

                                    if self.cfg.env in ['sawyer_peg_push']:
                                        temp_goal = np.zeros(3)
                                        temp_goal[0] = goal_coordinate[0]
                                        temp_goal[1] = goal_coordinate[1]
                                        temp_goal[2] = 0.01478
                                        goal_coordinate = temp_goal

                                        goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.05,0.05) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]
                                        goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.05,0.05) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]
                                        if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                            if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]:
                                                goal_coordinate[0] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] + 0.002
                                            if goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                                goal_coordinate[0] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0] - 0.002
                                        if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                            if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]:
                                                goal_coordinate[1] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] + 0.002
                                            if goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                                goal_coordinate[1] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1] - 0.002
                                    else:
                                        goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.5,0.5) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]
                                        goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.5,0.5) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]
                                        if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                            if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]:
                                                goal_coordinate[0] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] + 0.02
                                            if goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                                goal_coordinate[0] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0] - 0.02
                                        if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                            if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]:
                                                goal_coordinate[1] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] + 0.02
                                            if goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                                goal_coordinate[1] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1] - 0.02
                                    goal_coordinates.append(goal_coordinate)
                                    selected_images.append(f"output{i}.jpg")
                                i += 1
                            if not goal_coordinates:
                                print("No goals detected, running inference again.")
                        else:
                            output_imgs_xy = single_img((cur_image_no_goal*255).astype('uint8'),True, f"inpaint_results_xy/episode_{episode}/",model_pth = cur_model_dir_xy,beta_schedule_n_steps = self.cfg.beta_schedule_n_steps)
                            output_imgs_yz = single_img((cur_image_yz_no_goal*255).astype('uint8'),True, f"inpaint_results_yz/episode_{episode}/",model_pth = cur_model_dir_yz,beta_schedule_n_steps = self.cfg.beta_schedule_n_steps)
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
                                #goal_coordinate_yz = np.flip(goal_coordinate_yz)


                                goal_coordinate = np.zeros(3)
                                goal_coordinate[0] = goal_coordinate_xy[0]
                                goal_coordinate[1] = (goal_coordinate_xy[1] + goal_coordinate_yz[0])/2
                                goal_coordinate[2] = goal_coordinate_yz[1]

                                #print(goal_coordinate)

                                goal_coordinate[0] = goal_coordinate[0] + random.uniform(-0.05,0.05) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]
                                goal_coordinate[1] = goal_coordinate[1] + random.uniform(-0.05,0.05) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]
                                goal_coordinate[2] = goal_coordinate[2] + random.uniform(-0.01,0.01) + self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2]
                                if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] or goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                    if goal_coordinate[0] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0]:
                                        goal_coordinate[0] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[0] + 0.002
                                    if goal_coordinate[0] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0]:
                                        goal_coordinate[0] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[0] - 0.002
                                if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] or goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                    if goal_coordinate[1] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1]:
                                        goal_coordinate[1] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[1] + 0.002
                                    if goal_coordinate[1] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1]:
                                        goal_coordinate[1] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[1] - 0.002
                                if goal_coordinate[2] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2] or goal_coordinate[2] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[2]:
                                    if goal_coordinate[2] <= self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2]:
                                        goal_coordinate[2] = self.uniform_goal_sampler.LOWER_CONTEXT_BOUNDS[2] + 0.002
                                    if goal_coordinate[2] >= self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[2]:
                                        goal_coordinate[2] = self.uniform_goal_sampler.UPPER_CONTEXT_BOUNDS[2] - 0.002
    
                                goal_coordinates.append(goal_coordinate)
                                selected_images_xy.append(f"output{i}.jpg")
                                selected_images_yz.append(f"output{i}.jpg")
                                i += 1
                            if not goal_coordinates:
                                print("No goals detected, running inference again.")

                    #print("Goals are at", goal_coordinates)
                    #print("Selected images are", selected_images)

                    if not self.cfg.use_inpainting_aim and not self.cfg.use_inpainting_q:
                        if episode == 0:
                            print("Using mean of all goals")
                        goal_candidates = np.array(goal_coordinates)
                        inpaint_goal = np.mean(goal_candidates,axis=0)
                    elif self.cfg.use_inpainting_aim and not self.cfg.use_inpainting_q:
                        goal_candidates = np.array(goal_coordinates)
                        num_grid_point = goal_candidates.shape[0]
                        
                        if episode == 0: # sample random action for episode 0
                            print("Using aim for goal selection")

                        if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                            initial_states = np.tile(np.array([0.4, 0.8, 0.02]), (num_grid_point, 1))
                        else:
                            initial_states = np.tile(np.array([0,0]), (num_grid_point, 1))

                        observes = torch.as_tensor(np.concatenate([initial_states, goal_candidates], axis= -1), device = 0).float()
                        #print(observes.shape)
                        aim_output = agent.aim_discriminator.forward(observes).detach().cpu().numpy().flatten()
                        #print("Aim output is ", aim_output)

                        probabilities = softmax(-1*aim_output).flatten()
                        #print("Probabilities",probabilities)
                        optimal_goal = np.random.choice(aim_output, p = probabilities)
                        optimal_goal_index = np.argmax(aim_output==optimal_goal)

                        #print(f"For episode {episode} Optimal goal is output{optimal_goal_index}.jpg with coordinate {goal_candidates[optimal_goal_index]}")
                        if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                            os.rename(f"inpaint_results_xy/episode_{episode}/{selected_images_xy[optimal_goal_index]}",f"inpaint_results_xy/episode_{episode}/selected.jpg")
                            os.rename(f"inpaint_results_yz/episode_{episode}/{selected_images_yz[optimal_goal_index]}",f"inpaint_results_yz/episode_{episode}/selected.jpg")
                        else:
                            os.rename(f"inpaint_results/episode_{episode}/{selected_images[optimal_goal_index]}",f"inpaint_results/episode_{episode}/selected.jpg")
                        inpaint_goal = goal_candidates[optimal_goal_index]
                    elif self.cfg.use_inpainting_q and not self.cfg.use_inpainting_aim:
                        goal_candidates = np.array(goal_coordinates)
                        if self.cfg.use_inpainting_q_final: #Does not work well
                            cur_obs_list = []
                            cur_actions = []
                            if episode == 0: # sample random action for episode 0
                                print("Using Q with final action/obs")
                                spec = self.env.action_spec()
                                action = np.random.uniform(spec.low, spec.high, spec.shape)
                            #print(obs)
                            for goal_candidate in goal_candidates: 
                                my_obs = obs
                                if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                    my_obs[7:] = goal_candidate
                                else:
                                    my_obs[8:] = goal_candidate
                                #print(goal_candidate)
                                #print(my_obs)
                                #print(my_obs.shape)
                                if self.cfg.normalize_rl_obs:
                                    my_obs = agent.normalize_obs(my_obs,self.cfg.env)
                                cur_obs_list.append(my_obs)
                                cur_actions.append(action)
                            obs_arr = torch.as_tensor(np.stack(cur_obs_list,axis=0), device = 0).float()
                            action_arr = torch.as_tensor(np.stack(cur_actions,axis=0), device = 0).float()
                            
                            '''
                            print("**************************")
                            print(f"obs_arr is {obs_arr}\n\n\n")
                            print("**************************")
                            print(f"action_arr is {action_arr}\n\n\n")
                            print("**************************")
                            '''

                            inpaint_Q1,inpaint_Q2 = agent.critic(obs_arr,action_arr)
                            #print(inpaint_Q1,inpaint_Q2)

                            #print("Obs all shape: ", np.array(observe_list).shape)
                            #print("Action all shape: ", np.array(action_list).shape)

                            Q_means = ((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten()
                            probabilities = softmax(Q_means).flatten()
                            #print("Probabilities",probabilities)
                            optimal_goal = np.random.choice(Q_means, p = probabilities)
                            optimal_goal_index = np.argmax(Q_means==optimal_goal)
                            if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                os.rename(f"inpaint_results_xy/episode_{episode}/{selected_images_xy[optimal_goal_index]}",f"inpaint_results_xy/episode_{episode}/selected.jpg")
                                os.rename(f"inpaint_results_yz/episode_{episode}/{selected_images_yz[optimal_goal_index]}",f"inpaint_results_yz/episode_{episode}/selected.jpg")
                            else:
                                os.rename(f"inpaint_results/episode_{episode}/{selected_images[optimal_goal_index]}",f"inpaint_results/episode_{episode}/selected.jpg")
                            inpaint_goal = goal_candidates[optimal_goal_index]
                        elif self.cfg.use_inpainting_q_initial:
                            cur_obs_list = []
                            cur_actions = []
                            if episode == 0: # sample random action for episode 0
                                print("Using Q with initial action/obs")
                                init_obs = obs
                                spec = self.env.action_spec()
                                init_action = np.random.uniform(spec.low, spec.high, spec.shape)
                            #print(obs)
                            for goal_candidate in goal_candidates: 
                                my_obs = init_obs
                                if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                    my_obs[7:] = goal_candidate
                                else:
                                    my_obs[8:] = goal_candidate
                                #print(goal_candidate)
                                #print(my_obs)
                                #print(my_obs.shape)
                                if self.cfg.normalize_rl_obs:
                                    my_obs = agent.normalize_obs(my_obs,self.cfg.env)
                                cur_obs_list.append(my_obs)
                                cur_actions.append(init_action)
                            obs_arr = torch.as_tensor(np.stack(cur_obs_list,axis=0), device = 0).float()
                            action_arr = torch.as_tensor(np.stack(cur_actions,axis=0), device = 0).float()
                            
                            '''
                            print("**************************")
                            print(f"obs_arr is {obs_arr}\n\n\n")
                            print("**************************")
                            print(f"action_arr is {action_arr}\n\n\n")
                            print("**************************")
                            '''

                            inpaint_Q1,inpaint_Q2 = agent.critic(obs_arr,action_arr)
                            #print(inpaint_Q1,inpaint_Q2)

                            #print("Obs all shape: ", np.array(observe_list).shape)
                            #print("Action all shape: ", np.array(action_list).shape)

                            Q_means = ((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten()
                            probabilities = softmax(Q_means).flatten()
                            #print("Probabilities",probabilities)
                            optimal_goal = np.random.choice(Q_means, p = probabilities)
                            optimal_goal_index = np.argmax(Q_means==optimal_goal)
                            if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                os.rename(f"inpaint_results_xy/episode_{episode}/{selected_images_xy[optimal_goal_index]}",f"inpaint_results_xy/episode_{episode}/selected.jpg")
                                os.rename(f"inpaint_results_yz/episode_{episode}/{selected_images_yz[optimal_goal_index]}",f"inpaint_results_yz/episode_{episode}/selected.jpg")
                            else:
                                os.rename(f"inpaint_results/episode_{episode}/{selected_images[optimal_goal_index]}",f"inpaint_results/episode_{episode}/selected.jpg")
                            inpaint_goal = goal_candidates[optimal_goal_index]

                        elif self.cfg.use_inpainting_q_all:
                            q_mean_list = []
                            if episode == 0: # Episode 0 is an edge case as there is only 1 observation and no action
                                print("Using Q with all action/obs")
                                spec = self.env.action_spec()
                                action = np.random.uniform(spec.low, spec.high, spec.shape)
                            #print(obs)
                                cur_obs_list = []
                                cur_actions = []
                                for goal_candidate in goal_candidates: 
                                    my_obs = obs
                                    if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                        my_obs[7:] = goal_candidate
                                    else:
                                        my_obs[8:] = goal_candidate

                                    if self.cfg.normalize_rl_obs:
                                        my_obs = agent.normalize_obs(my_obs,self.cfg.env)
                                    cur_obs_list.append(my_obs)
                                    cur_actions.append(action)
                                obs_arr = torch.as_tensor(np.stack(cur_obs_list,axis=0), device = 0).float()
                                action_arr = torch.as_tensor(np.stack(cur_actions,axis=0), device = 0).float()
                                inpaint_Q1,inpaint_Q2 = agent.critic(obs_arr,action_arr)

                                Q_means = ((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten()
                                #print(Q_means.shape) # (n,)
                            else:
                                observe_array = np.array(observe_list)
                                action_array = np.array(action_list)

                                action_array = torch.as_tensor(np.stack(action_array,axis=0), device = 0).float()
                                for goal_candidate in goal_candidates: 
                                    cur_observes = np.array(observe_array)
                                    for item in cur_observes: #update cur_observes with the current goal candidate
                                        if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                            item[7:] = goal_candidate
                                        else:
                                            item[8:] = goal_candidate
                                        if self.cfg.normalize_rl_obs:
                                            item = agent.normalize_obs(item,self.cfg.env)
                                    cur_observes = torch.as_tensor(np.stack(cur_observes,axis=0), device = 0).float()
                                    inpaint_Q1,inpaint_Q2 = agent.critic(cur_observes,action_array)
                                    cur_q_mean = np.mean(((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten())
                                    #print(cur_q_mean)
                                    q_mean_list.append(cur_q_mean)
                                Q_means = np.array(q_mean_list)
                                #print(Q_means.shape) # (n,)

                            probabilities = softmax(Q_means).flatten()
                            #print("Probabilities",probabilities)
                            optimal_goal = np.random.choice(Q_means, p = probabilities)
                            optimal_goal_index = np.argmax(Q_means==optimal_goal)
                            if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                os.rename(f"inpaint_results_xy/episode_{episode}/{selected_images_xy[optimal_goal_index]}",f"inpaint_results_xy/episode_{episode}/selected.jpg")
                                os.rename(f"inpaint_results_yz/episode_{episode}/{selected_images_yz[optimal_goal_index]}",f"inpaint_results_yz/episode_{episode}/selected.jpg")
                            else:
                                os.rename(f"inpaint_results/episode_{episode}/{selected_images[optimal_goal_index]}",f"inpaint_results/episode_{episode}/selected.jpg")
                            inpaint_goal = goal_candidates[optimal_goal_index]
                        
                    elif self.cfg.use_inpainting_q and self.cfg.use_inpainting_aim:
                        q_mean_list = []
                        goal_candidates = np.array(goal_coordinates)

                        #get q values
                        if episode == 0: # Episode 0 is an edge case as there is only 1 observation and no action
                            print("Using Q and aim")
                            spec = self.env.action_spec()                
                            action = np.random.uniform(spec.low, spec.high, spec.shape)
                            #print(obs)
                            cur_obs_list = []
                            cur_actions = []
                            for goal_candidate in goal_candidates: 
                                my_obs = obs
                                if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                    my_obs[7:] = goal_candidate
                                else:
                                    my_obs[8:] = goal_candidate

                                if self.cfg.normalize_rl_obs:
                                    my_obs = agent.normalize_obs(my_obs,self.cfg.env)
                                cur_obs_list.append(my_obs)
                                cur_actions.append(action)
                                #print("My obs is",my_obs)
                            obs_arr = torch.as_tensor(np.stack(cur_obs_list,axis=0), device = 0).float()
                            action_arr = torch.as_tensor(np.stack(cur_actions,axis=0), device = 0).float()
                            inpaint_Q1,inpaint_Q2 = agent.critic(obs_arr,action_arr)

                            Q_means = ((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten()
                            #print(Q_means.shape) # (n,)
                        else:
                            observe_array = np.array(observe_list)
                            action_array = np.array(action_list)

                            action_array = torch.as_tensor(np.stack(action_array,axis=0), device = 0).float()
                            for goal_candidate in goal_candidates: 
                                cur_observes = np.array(observe_array)
                                for item in cur_observes: #update cur_observes with the current goal candidate
                                    if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                                        item[7:] = goal_candidate
                                    else:
                                        item[8:] = goal_candidate
                                    if self.cfg.normalize_rl_obs:
                                        item = agent.normalize_obs(item,self.cfg.env)
                                #print(item)
                                cur_observes = torch.as_tensor(np.stack(cur_observes,axis=0), device = 0).float()
                                inpaint_Q1,inpaint_Q2 = agent.critic(cur_observes,action_array)
                                cur_q_mean = np.mean(((inpaint_Q1 + inpaint_Q2)/2).detach().cpu().numpy().flatten())
                                #print(cur_q_mean)
                                q_mean_list.append(cur_q_mean)
                            Q_means = np.array(q_mean_list)
                            #print(Q_means.shape) # (n,)

                        #Get aim rewards
                        num_grid_point = goal_candidates.shape[0]

                        if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                            initial_states = np.tile(np.array([0.4, 0.8, 0.02]), (num_grid_point, 1))
                        else:
                            initial_states = np.tile(np.array([0,0]), (num_grid_point, 1))

                        observes = torch.as_tensor(np.concatenate([initial_states, goal_candidates], axis= -1), device = 0).float()
                        #print(observes.shape)
                        aim_output = agent.aim_discriminator.forward(observes).detach().cpu().numpy().flatten()
                        #print("Aim output is ", aim_output)

                        merge_q_mean_aim = Q_means*10 + ((-1)*aim_output)

                        probabilities = softmax(merge_q_mean_aim).flatten()
                        #print("Probabilities",probabilities)
                        optimal_goal = np.random.choice(Q_means, p = probabilities)
                        optimal_goal_index = np.argmax(Q_means==optimal_goal)
                        if self.cfg.env in ['sawyer_peg_push'] or self.cfg.env in ['sawyer_peg_pick_and_place']:
                            os.rename(f"inpaint_results_xy/episode_{episode}/{selected_images_xy[optimal_goal_index]}",f"inpaint_results_xy/episode_{episode}/selected.jpg")
                            os.rename(f"inpaint_results_yz/episode_{episode}/{selected_images_yz[optimal_goal_index]}",f"inpaint_results_yz/episode_{episode}/selected.jpg")
                        else:
                            os.rename(f"inpaint_results/episode_{episode}/{selected_images[optimal_goal_index]}",f"inpaint_results/episode_{episode}/selected.jpg")
                        inpaint_goal = goal_candidates[optimal_goal_index]


                    #use obs, change 8-10 (desired goal), use final action obs 1x10 şeklinde
                    #obs = self.normalize_obs(obs, self.env_name) sel.agent ın içinde herhalde normalize et obs yi normalize_rl_obs ye göre yap
                    #
                    #action > son aciton burada
                    #Q1,Q2 = agent.critic(normalized_obs,action) state de action da final action 
                    #                                                                   YA DA obs_0 ı alabilrim o da goal setlediğim yerde
                    #                                                                    ve action için actor(obs_0) action veriyor o da action_0 veriyor outpacesacde 
                    # sonuç Q1, Q2 ikisi de float meanleri alabilirim ya da tekini kullan
                    #burada ne kadar pozitif o kadar iyi

                    # 3. option tüm action ve obs leri buffer a koy, u maze için 100 lük buffer olmalı, sonra goalları replace le, normalize et, q1,q2 .çıkart
                    # bunda 1 batchede veremeyeceksin 100x1 lik sonuç gelecek her goal için diğerinde 1x1 sonu gelecek her goal için

                    #q ve reward ikisini de tutup deneyebilirim, e.g. x*q + y*aim_reward maximize this x = 10, y = 1 denenebilir.

                positions = []
                action_list = []
                observe_list = []


                episode_reward = 0
                episode_step = 0
                episode += 1
                episode_observes = [obs]
                self.logger.log('train/episode', episode, self.step)

            agent = self.get_agent()
            replay_buffer = self.get_buffer()
            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                print('eval started...')
                self.logger.log('eval/episode', episode - 1, self.step)
                self.evaluate(eval_uniform_goal=False)                

                if self.step > self.cfg.num_random_steps:
                    temp_obs, _, _, _, _, _ = self.aim_expl_buffer.sample_without_relabeling(128, agent.discount, sample_only_state = False)
                    temp_obs = temp_obs.detach().cpu().numpy()
                    temp_obs_dict = self.env.convert_obs_to_dict(temp_obs)
                     
                    temp_dg = temp_obs_dict['desired_goal']
                    
                    fig = plt.figure()
                    sns.set_style("darkgrid")
                    
                    ax1 = fig.add_subplot(1,1,1)                                    
                    ax1.scatter(temp_dg[:, 0], temp_dg[:, 1], label = 'goals')
                              
                    if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
                        x_min, x_max = -2, 10
                        y_min, y_max = -2, 10
                    elif self.cfg.env == "PointSpiralMaze-v0":
                        x_min, x_max = -10, 10
                        y_min, y_max = -10, 10
                    elif self.cfg.env in ["PointNMaze-v0"]:
                        x_min, x_max = -2, 10
                        y_min, y_max = -2, 18
                    elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                        x_min, x_max = -0.6, 0.6
                        y_min, y_max = 0.2, 1.0
                    else:
                        raise NotImplementedError

                    plt.xlim(x_min,x_max)    
                    plt.ylim(y_min,y_max)
                    
                                  

                    ax1.legend(loc ="best") # 'upper right' # , prop={'size': 20}          
                    plt.savefig(self.eval_video_recorder.save_dir+'/curriculum_goals_'+str(self.step)+'.jpg')
                    plt.close()
                
                if self.cfg.use_residual_randomwalk and (self.randomwalk_buffer.idx > 128 or self.randomwalk_buffer.full):
                    temp_obs, _, _, _, _, _ = self.randomwalk_buffer.sample_without_relabeling(128, agent.discount, sample_only_state = False)
                    temp_obs = temp_obs.detach().cpu().numpy()
                    temp_obs_dict = self.env.convert_obs_to_dict(temp_obs)
                     
                    temp_dg = temp_obs_dict['desired_goal']
                    temp_ag = temp_obs_dict['achieved_goal']
                    
                    fig = plt.figure()
                    sns.set_style("darkgrid")
                    
                    ax1 = fig.add_subplot(1,1,1)                                    
                    ax1.scatter(temp_dg[:, 0], temp_dg[:, 1], label = 'goals')
                    ax1.scatter(temp_ag[:, 0], temp_ag[:, 1], label = 'achieved states', color = 'red')
                              
                    if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0"]:
                        x_min, x_max = -2, 10
                        y_min, y_max = -2, 10
                    elif self.cfg.env == "PointSpiralMaze-v0":
                        x_min, x_max = -10, 10
                        y_min, y_max = -10, 10
                    elif self.cfg.env in ["PointNMaze-v0"]:
                        x_min, x_max = -2, 10
                        y_min, y_max = -2, 18
                    elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                        x_min, x_max = -0.6, 0.6
                        y_min, y_max = 0.2, 1.0
                    else:
                        raise NotImplementedError
                    plt.xlim(x_min,x_max)    
                    plt.ylim(y_min,y_max)
                    
                    

                    ax1.legend(loc ="best") # 'upper right' # , prop={'size': 20}          
                    plt.savefig(self.eval_video_recorder.save_dir+'/randomwalk_goalandstates_'+str(self.step)+'.jpg')
                    plt.close()
                    

            #print(f"Episode: {episode}, Step {self.step}, goal {self.env.goal}")

            # save agent periodically
            if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
                utils.save(
                    self.expl_agent,
                    os.path.join(self.model_dir, f'expl_agent_{self.step}.pt'))                
            if self.cfg.save_buffer and (self.step % self.cfg.buffer_save_frequency == 0) :
                utils.save(self.expl_buffer.replay_buffer, os.path.join(self.buffer_dir, f'buffer_{self.step}.pt'))
                utils.save(self.aim_expl_buffer.replay_buffer, os.path.join(self.buffer_dir, f'aim_disc_buffer_{self.step}.pt'))
                if self.cfg.use_residual_randomwalk:
                    utils.save(self.randomwalk_buffer.replay_buffer, os.path.join(self.buffer_dir, f'randomwalk_buffer_{self.step}.pt'))
            
                if self.cfg.use_hgg:
                    utils.save(self.hgg_achieved_trajectory_pool,  os.path.join(self.buffer_dir, f'hgg_achieved_trajectory_pool_{self.step}.pt'))
                

            # sample action for data collection
            if self.step < self.cfg.num_random_steps or (self.cfg.randomwalk_method == 'rand_action' and self.env.is_residual_goal): #This is movement
                spec = self.env.action_spec()                
                action = np.random.uniform(spec.low, spec.high,
                                        spec.shape)
                
            else: 
                with utils.eval_mode(agent):
                    action = agent.act(obs, spec = self.env.action_spec(), sample=True)
            
            logging_dict = agent.update(replay_buffer, self.randomwalk_buffer, self.aim_expl_buffer, self.step, self.env, self.goal_buffer)
            
            if self.step % self.cfg.logging_frequency== 0:                
                if logging_dict is not None: # when step = 0                                        
                    for key, val in logging_dict.items():
                        self.logger.log('train/'+key, val, self.step)
            

            #numpy_array_container.append(obs)

            next_obs, reward, done, info = self.env.step(action)
            
            #self.env.render()

            episode_reward += reward
            episode_observes.append(next_obs)

            last_timestep = True if (episode_step+1) % self.max_episode_timesteps == 0 or done else False


            
            self.train_video_recorder.record(self.env)



            if self.cfg.use_residual_randomwalk:
                if self.env.is_residual_goal:
                    self.randomwalk_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
                else:
                    replay_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
                    self.aim_expl_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)

            else:
                replay_buffer.add(obs, action, reward, next_obs, done, last_timestep)
                self.aim_expl_buffer.add(obs, action, reward, next_obs, done, last_timestep)

                
            if last_timestep:
                replay_buffer.add_trajectory(episode_observes)
                replay_buffer.store_episode()
                self.aim_expl_buffer.store_episode()
                if self.randomwalk_buffer is not None:
                    self.randomwalk_buffer.store_episode()
                if self.randomwalk_buffer is not None:
                    if (not replay_buffer.full) and (not self.randomwalk_buffer.full):
                        assert self.step+1 == self.randomwalk_buffer.idx + replay_buffer.idx
                else:
                    if not replay_buffer.full:
                        assert self.step+1 == replay_buffer.idx

                if self.cfg.use_hgg:                    
                    temp_episode_observes = copy.deepcopy(episode_observes)
                    temp_episode_ag = []                                        
                     # NOTE : should it be [obs, ag] ?
                    if 'aim_f' in self.hgg_sampler.cost_type or 'meta_nml' in self.hgg_sampler.cost_type:
                        temp_episode_init = self.eval_env.convert_obs_to_dict(temp_episode_observes[0])['achieved_goal'] # for bias computing
                    else:    
                        raise NotImplementedError
                        

                    for k in range(len(temp_episode_observes)):
                        temp_episode_ag.append(self.eval_env.convert_obs_to_dict(temp_episode_observes[k])['achieved_goal'])
                    
                    if getattr(self.env, 'full_state_goal', False):
                        raise NotImplementedError("You should modify the code when full_state_goal (should address achieved_goal to compute goal distance below)")


                    achieved_trajectories = [np.array(temp_episode_ag)] # list of [ts, dim]
                    achieved_init_states = [temp_episode_init] # list of [ts(1), dim]

                    selection_trajectory_idx = {}
                    for i in range(len(achieved_trajectories)):                                                 
                        # full state achieved_goal
                        if self.cfg.env in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                            threshold = 0.2
                        elif self.cfg.env in [     'sawyer_peg_push','sawyer_peg_pick_and_place']:
                            threshold = 0.02
                        else:
                            raise NotImplementedError
                        if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>threshold: # if there is a difference btw first and last timestep ?
                            selection_trajectory_idx[i] = True
                    
                    hgg_achieved_trajectory_pool = self.hgg_achieved_trajectory_pool
                    for idx in selection_trajectory_idx.keys():
                        hgg_achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy()) # if trajectory is long put it into the trajectory pool
                    
                    
                    
            #print(info)
            positions.append(info["position"])
            observe_list.append(obs)
            action_list.append(action)

            if episode_step == 0:
                init_obs = obs
                init_action = action

            obs = next_obs
            episode_step += 1
            self.step += 1
            
            if self.cfg.use_residual_randomwalk:
                if self.env.is_residual_goal:
                    if (self.env.residual_goalstep % 10 == 0) or info.get('is_current_goal_success'):
                        if (self.cfg.use_uncertainty_for_randomwalk not in [None, 'none', 'None']) and self.step > self.get_agent().meta_test_sample_size:
                            residual_goal = self.get_agent().sample_randomwalk_goals(obs = obs, ag = self.env.convert_obs_to_dict(obs)['achieved_goal'], \
                                episode = episode, env=self.env, replay_buffer = self.get_inv_weight_curriculum_buffer(), \
                                num_candidate = self.cfg.randomwalk_num_candidate, random_noise = self.cfg.randomwalk_random_noise, \
                                uncertainty_mode = self.cfg.use_uncertainty_for_randomwalk)
                        else:
                            noise = np.random.uniform(low=-self.cfg.randomwalk_random_noise, high=self.cfg.randomwalk_random_noise, size=self.env.goal_dim)
                            
                            if self.cfg.env in [   'sawyer_peg_pick_and_place']:
                                assert self.cfg.randomwalk_random_noise <= 0.2
                                pass
                            elif self.cfg.env in [  'sawyer_peg_push']:
                                assert self.cfg.randomwalk_random_noise <= 0.2
                                noise[2] = 0
                            residual_goal = self.env.convert_obs_to_dict(obs)['achieved_goal'] + noise
                            
                        self.env.reset_goal(residual_goal)
                        obs[-self.env.goal_dim:] = residual_goal.copy()
                else:
                    if info.get('is_current_goal_success'): #succeed original goal
                        self.env.original_goal_success = True
                        if (self.cfg.use_uncertainty_for_randomwalk not in [None, 'none', 'None']) and self.step > self.get_agent().meta_test_sample_size:
                            residual_goal = self.get_agent().sample_randomwalk_goals(obs = obs, ag = self.env.convert_obs_to_dict(obs)['achieved_goal'], \
                                episode = episode, env=self.env, replay_buffer = self.get_inv_weight_curriculum_buffer(), \
                                num_candidate = self.cfg.randomwalk_num_candidate, random_noise = self.cfg.randomwalk_random_noise, \
                                uncertainty_mode = self.cfg.use_uncertainty_for_randomwalk)
                        else:
                            noise = np.random.uniform(low=-self.cfg.randomwalk_random_noise, high=self.cfg.randomwalk_random_noise, size=self.env.goal_dim)

                            if self.cfg.env in [   'sawyer_peg_pick_and_place']:
                                assert self.cfg.randomwalk_random_noise <= 0.2
                                pass
                            elif self.cfg.env in [  'sawyer_peg_push']:
                                assert self.cfg.randomwalk_random_noise <= 0.2
                                noise[2] = 0
                            residual_goal = self.env.convert_obs_to_dict(obs)['achieved_goal'] + noise
                        self.env.reset_goal(residual_goal)
                        obs[-self.env.goal_dim:] = residual_goal.copy()
                if (episode_step) % self.max_episode_timesteps == 0: #done only horizon ends
                    done = True
                    info['is_success'] = self.env.original_goal_success

                    if self.cfg.trajectory_save:
                        np.save(self.trajectory_dir+ "/" +str(self.step)+".npy",np.array(numpy_array_container))
                        np.save(self.trajectory_dir + "/generated_curriculum_points" + str(self.step) + ".npy", np.array(generated_curriculum_points))
                        numpy_array_container = []
                        generated_curriculum_points = []

@hydra.main(config_path='./config', config_name='config_outpace.yaml')
def main(cfg):
    import os
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    from outpace_train import Workspace as W
    

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()