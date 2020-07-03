import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
import gym
from gym.spaces import Box,Discrete
from torch.optim import Adam
import time
import scipy.signal

'''目前的错误。buffer那里还没有存满'''

#实现AC的基础，critic总是为值函数，actor分针对离散和连续变量
def combined_shape(length,dimension=None):
    '''这是个辅助函数，length代表了buffer里面存的个数，dimension代表了动作空间或观测空间的维度'''
    if dimension is None:
        return (length,)
    return (length,dimension) if np.isscalar(dimension) else (length,*dimension) 
def creat_network(sizes,activation,output_activation=nn.Identity):
    '''sizes代表着一个列表，列表中存在网络的输入和输出的维度，根据此来确定网络结构'''
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
class Critic(nn.Module):
    '''一个critic应当实现什么样的功能？
    首先是传入state，然后输出值函数，把建立模型这个模块剥离出来'''
    def __init__(self,observation_dim,hidden_sizes,activation):
        super().__init__()
        self.value_function=creat_network([observation_dim]+list(hidden_sizes)+[1],activation)
    def forward(self,observation):
        value=self.value_function(observation)
        return torch.squeeze(value,-1)#迫使其维度为1，从而符合数据格式要求

class Actor(nn.Module):

    '''actor功能应该是什么？，输入state，输出action'''
    def forward(self,observation,action=None):
        #1.本质是输出动作，然而实质是通过输出distribution of action 和logprob_a来实现这个事情
        policy=self._distribution(observation)
        logprob_action=None
        if action is not None:
            logprob_action=self._logprob_from_distribution(policy,action)
        return policy, logprob_action 
    def _distribution(self,observation):
        raise NotImplementedError
    def _logprob_from_distribution(self,action,policy):
        raise NotImplementedError
class Actor_Discrete(Actor):

    def __init__(self,observation_dim,action_dim,hidden_sizes,activation):
        super().__init__()
        self.actor_network=creat_network([observation_dim]+list(hidden_sizes)+[action_dim],activation)
    def _distribution(self, observation):
        '''输出一个返回的是一个policy，即一个输出动作的概率分布，这种分布就是policy'''
        logits=self.actor_network(observation)#执行update时此刻输入为4000*4，输出为4000*2
        return Categorical(logits=logits)
    def _logprob_from_distribution(self,policy,action):
        '''输出一个policy采取某一动作的logprob'''
        logprob_action=policy.log_prob(action)
        return logprob_action

class Actor_Continue(Actor):

    def __init__(self,observation_dim,action_dim,hidden_sizes,activation):
        super().__init__()
        log_std=-0.5*torch.ones(action_dim,dtype=torch.float32)
        self.log_std=torch.nn.Parameter(log_std)#表征一个分布的方差，并将其绑定为参数，因而可优化
        self.actor_network=creat_network([observation_dim]+list(hidden_sizes)+[action_dim],activation)#关键在此
    def _distribution(self,observation):
        '''输出一个返回的是一个policy，连续型动作输出的应是高斯分布，表征其分布必然就需要两类数据，方差和均值'''
        mu=self.actor_network(observation)#均值
        std=torch.exp(self.log_std)
        
        return Normal(mu,std)
    def _logprob_from_distribution(self,policy,action):
        '''输出一个policy采取某一动作的logprob'''
        logprob_action=policy.log_prob(action).sum(axis=-1)
        return logprob_action
class ActorCritic(nn.Module):
    '''所需传入初始化的参数有observation space, action space, hidden_size=(64,64)，activiation=nn.Tanh
    所需外部提供，且用于函数计算的参数为,observation'''
    def __init__(self, observation_space, action_space,hidden_sizes=(64,64), 
                  activation=nn.Tanh):
        super().__init__()

        observation_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):#连续动作空间
            self.policy_function = Actor_Continue(observation_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):#for离散动作空间
            self.policy_function = Actor_Discrete(observation_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.value_function  = Critic(observation_dim, hidden_sizes, activation)

    def step(self, observation):
        '''所返回的参数为numpy格式的action,value,logprob_a'''
        with torch.no_grad():
            policy = self.policy_function._distribution(observation)
            action = policy.sample()
            logprob_a = self.policy_function._logprob_from_distribution(policy, action)#此处需强调下policy_function 和policy的区别
            value = self.value_function(observation)
        return action.numpy(), value.numpy(), logprob_a.numpy()

    def act(self, observation):
        '''所返回的参数为numpy 格式的action'''
        return self.step(observation)[0]
class Buffer:#假装是experience relay，实则是一个普通的on-policy过程存储，集成了GAE功能哦
    """
   假装是experience relay，实则是一个普通的on-policy过程存储，集成了GAE功能哦.
    """
    #利用列表来实现buffe
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1#计步器

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        def _discount(x,discount_rate):
            return scipy.signal.lfilter([1], [1, float(-discount_rate)], x[::-1], axis=0)[::-1]
        path_slice = slice(self.path_start_idx, self.ptr)#一个episode
        rews = np.append(self.rew_buf[path_slice], last_val)#一个episode的reward，并添加last_val至array中
        vals = np.append(self.val_buf[path_slice], last_val)#一个episode的value，并添加last_val至array中
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]#优势函数计算，delta=rt+(vt+1)-vt
        self.adv_buf[path_slice] = _discount(deltas, self.gamma * self.lam)#GAE形式的优势函数
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = _discount(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr#将记步进行到底

    def get(self):#归一化字典并返回torch
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(),self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
# a test for the model 
#好戏开场了
gamma=0.99
number_epoches=2
lam=0.98
# epochs=100
policy_lr=3e-4
value_lr=2e-3
max_steps_per_epoch=4000
seed=1000
torch.manual_seed(seed)
np.random.seed(seed)
env=gym.make('CartPole-v0')
observation_dimension=env.observation_space.shape
action_dimension=env.action_space.shape
actor_critic=ActorCritic(env.observation_space,env.action_space)
buffer=Buffer(observation_dimension,action_dimension,max_steps_per_epoch,gamma,lam)#实例化一个缓存空间

def train(agent,buffer,env):
    '''agent 就是前文创建的actor_critic，buffer就是数据来源，env就是环境'''
    def calculate_loss(data):
        observation,action,returned,advantage,logprob=data['obs'],data['act'],data['ret'], data['adv'], data['logp']
        loss_value=((agent.value_function(observation)-returned)**2).mean()
        policy, logprob =agent.policy_function(observation,action)
        loss_policy=-(logprob*advantage).mean()
        return loss_policy,loss_value
    def update():
        '''更新参数的标准流程'''
        data=buffer.get()
        loss_policy,loss_value=calculate_loss(data)
        # loss_policy=loss_policy.item()
        # loss_value=loss_value.item()
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss_policy.backward()
        loss_value.backward()
        policy_optimizer.step()
        value_optimizer.step()
    #设定好optim函数
    policy_optimizer=Adam(agent.policy_function.parameters(),lr=policy_lr)
    value_optimizer=Adam(agent.value_function.parameters(),lr=value_lr)
    start_time=time.time()
    observation,episode_returned,episode_length=env.reset(),0,0
    #####开始啦
    for epoch in range(number_epoches):
        for t in range(max_steps_per_epoch):
            action,value,logprob_a=agent.step(torch.as_tensor(observation,dtype=torch.float32))
            next_observation, reward, d, _ =env.step(action)
            episode_returned+=reward
            episode_length+=1
            buffer.store(observation,action,reward,value,logprob_a)
            observation=next_observation
            #判断终止条件
            # timeout=epoch_length==max_steps_per_epoch
            epoch_end= t==max_steps_per_epoch-1
            terminal=d
            if terminal or epoch_end:
                if epoch_end and not terminal:
                    _,value,_=agent.step(torch.as_tensor(observation,dtype=torch.float32))
        
                    print('一个episode被斩断了，风暴需要继续进行')
                else:
                    value=0
                buffer.finish_path(value)
                observation,epoch_returned,epoch_length=env.reset(),0,0#重置，为下一个episode做准备了
        update()
#主函数编写
if __name__=='__main__':
    train(actor_critic,buffer,env)

    






    

    