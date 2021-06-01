import gym
import numpy as numpy
import ic3net_envs

class PPWrapperEnv(gym.Env):

    def __init__(self,):
        self.env = None
        self.num_controlled_agents = 0
        self.action_space = None
        self.observation_space = None

    def init_args(self, parser):
        env = parser.add_argument_group('Multi-task PP')    
        env.add_argument('--num_controlled_agents', nargs='+', type=int, required=True,
                         help="the number of controlled agents for each task") 
        env_pp = gym.make('PredatorPrey-v0')
        env_pp.init_args(parser)

    def multi_agent_init(self, args):
        self.envs = []
        for i in range(len(args.num_controlled_agents)):
            args.nfriendly = args.num_controlled_agents[i]
            env = gym.make('PredatorPrey-v0')
            env.multi_agent_init(args)
            self.envs.append(env)

        self.cur_env = self.envs[0]
        self.nagents_list = args.num_controlled_agents
        self.cur_nagents = self.nagents_list[0]
        self.cur_idx = 0

        self.action_space = self.cur_env.action_space
        self.observation_space = self.cur_env.observation_space

        return

    def reset(self):
        self.stat = dict()
        return self.cur_env.reset()

    def step(self, actions):
        obs, rwd, episode_over, debug = self.cur_env.step(actions)
        stat_name = 'task%i_success' % (self.cur_idx)
        self.stat[stat_name] = self.cur_env.stat['success']
        return obs, rwd, episode_over, debug

    def seed(self):
        return
        
    def change_env(self):
        if self.cur_idx == len(self.envs)-1:
            self.cur_idx = 0
        else:
            self.cur_idx += 1

        self.cur_env = self.envs[self.cur_idx]
        self.cur_nagents = self.nagents_list[self.cur_idx]



