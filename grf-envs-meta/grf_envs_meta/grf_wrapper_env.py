import gym
import gfootball.env as grf_env
import numpy as np

class GRFWrapperEnv(gym.Env):
    
    def __init__(self,):
        self.env = None
        self.num_controlled_agents = 0
        self.num_lagents = 0
        self.num_ragents = 0
        self.action_space = None
        self.observation_space = None
        
    def init_args(self, parser):
        env = parser.add_argument_group('GRF')
        env.add_argument('--scenarios', nargs='+', required=True,
                         help="Scenarios (tasks) of the game")      
        env.add_argument('--num_controlled_agents', nargs='+', type=int, required=True,
                         help="the number of controlled agents for each task")  
        env.add_argument('--max_num_lplayers', type=int, default=4,
                         help="Max number of players on the left side")
        env.add_argument('--max_num_rplayers', type=int, default=3,
                         help="Max number of players on the right side")
        env.add_argument('--reward_type', type=str, default='scoring',
                         help="Reward type for training")
        env.add_argument('--render', action="store_true", default=False,
                         help="Render training or testing process")
        
    def multi_agent_init(self, args):
        self.envs = []
        for i in range(len(args.scenarios)):
            env = grf_env.create_environment(
                env_name=args.scenarios[i],
                stacked=False,
                representation='multiagent',
                rewards=args.reward_type,
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                dump_frequency=0,
                logdir='/tmp/test',
                extra_players=None,
                number_of_left_players_agent_controls=args.num_controlled_agents[i],
                number_of_right_players_agent_controls=0,
                channel_dimensions=(3, 3))

            self.envs.append(env)

        self.cur_env = self.envs[0]
        self.nagents_list = args.num_controlled_agents
        self.cur_nagents = self.nagents_list[0]
        self.cur_idx = 0

        self.render = args.render

        self.max_num_lplayers = args.max_num_lplayers # max number of players on the left team
        self.max_num_rplayers = args.max_num_rplayers # max number of players on the left team

        # the original action spaces for different tasks should be the same 
        if self.cur_nagents > 1:
            self.action_space = gym.spaces.Discrete(self.cur_env.action_space.nvec[1])
        else:
            self.action_space = self.cur_env.action_space

        self.max_num_players = self.max_num_lplayers + self.max_num_rplayers
        shape = 2 * self.max_num_players + 3
        self.observation_space = gym.spaces.Box(
                low=np.array([-1]*shape), 
                high=np.array([1]*shape), 
                shape=(shape, ), 
                dtype=self.cur_env.observation_space.dtype)
            
        return
   
    # check epoch arg
    def reset(self):
        if self.render:
            self.cur_env.render()
        self.stat = dict()
        obs = self.cur_env.reset()
        if self.cur_nagents == 1:
            obs = obs.reshape(1, -1)

        left_sup = np.zeros((obs.shape[0], 2*(self.max_num_lplayers-self.cur_env.num_lteam_players)))
        right_sup = np.zeros((obs.shape[0], 2*(self.max_num_rplayers-self.cur_env.num_rteam_players)))

        obs = np.concatenate((obs[:,0:2*self.cur_env.num_lteam_players], left_sup,
         obs[:,2*self.cur_env.num_lteam_players:2*(self.cur_env.num_lteam_players+self.cur_env.num_rteam_players)],
         right_sup, obs[:,-13:-10]), axis=1)

        return obs
    
    def step(self, actions):
        o, r, d, i = self.cur_env.step(actions)
        if self.cur_nagents == 1:
            o = o.reshape(1, -1)
            r = r.reshape(1, -1)

        left_sup = np.zeros((o.shape[0], 2*(self.max_num_lplayers-self.cur_env.num_lteam_players)))
        right_sup = np.zeros((o.shape[0], 2*(self.max_num_rplayers-self.cur_env.num_rteam_players)))

        o = np.concatenate((o[:,0:2*self.cur_env.num_lteam_players], left_sup,
         o[:,2*self.cur_env.num_lteam_players:2*(self.cur_env.num_lteam_players+self.cur_env.num_rteam_players)],
         right_sup, o[:,-13:-10]), axis=1)

        next_obs = o
        rewards = r
        dones = d
        infos = i

        stat_name = 'task%i_success' % (self.cur_idx)
        self.stat[stat_name] = infos['score_reward']
        
        return next_obs, rewards, dones, infos
        
    def seed(self):
        return
    
    def render(self):
        self.env.render()
        
    def exit_render(self):
        self.env.disable_render()

    def change_env(self):
        if self.render:
            self.cur_env.disable_render()

        if self.cur_idx == len(self.envs)-1:
            self.cur_idx = 0
        else:
            self.cur_idx += 1

        self.cur_env = self.envs[self.cur_idx]
        self.cur_nagents = self.nagents_list[self.cur_idx]
        
