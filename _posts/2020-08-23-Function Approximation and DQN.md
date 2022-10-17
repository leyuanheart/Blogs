---
layout:     post
title:      Reinforcement Learning 4
subtitle:   Function Approximation and DQN
date:       2020-08-22
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---


上一篇我们介绍了model-free的MDP方法，主要是MC和TD，之前举的例子都是离散动作和离散状态的环境，首先是因为解释起来很方便，其次，这些问题的状态和动作都比较少，因此我们可以通过维护一张Q table来存储我们想要的value值，但是现实世界中的很多问题状态和动作空间都非常大，比如

- 国际象棋：$10^{47}$ states
- 围棋：$10^{170}$ states
- 机器人控制：连续的状态空间

如果还用Q table的形式来表示，首先就会遇到存储容量不足的问题；其次，表中的每一个格子都需要单独来学习，这样会非常耗时间。所以，需要将原来的方法给扩展到大规模的RL问题（scaling up RL），做法就是**函数近似（Function Approximation）**。

在强化学习中有哪些内容是可以用函数近似的呢？主要有3类：

- 环境模型（转移概率和奖励函数）
- state value function和state-action value function
- 策略

$$
\begin{gather}
\hat{v}(s;w) \approx v^{\pi}(s) \\
\hat{q}(s,a;w) \approx q^{\pi}(s,a) \\
\hat{\pi}(a,s;w) \approx \pi(a|s)
\end{gather}
$$

函数近似的通过把要估计的值函数参数化，可以将结果**泛化**到一些没有见到过的状态和动作，但缺点就是在原来只是估计可能存在误差（Estimation error）的基础上又引入了近似误差（Approximation error）。

这次主要聚焦于value函数的的近似，这里给出一些函数的设计：

![27](https://pic.downk.cc/item/5f4139f0160a154a67d0bb32.png)

也有很多函数的形式可供选择，比如线性函数、决策树、神经网络等等，因为之后都是要介绍结合深度学习的RL，所以就只用神经网络了，更新参数的方式是利用梯度下降。

## Function Approximation with an Oracle

现在假设我们已知真实的value函数$v^{\pi}(s)$，我们该如何做函数近似呢，这其实就和常规的监督学习的流程一样了，先定义一个损失函数，比如mean square error：


$$
J(w)=E_{\pi}[(v^{\pi}(s)-\hat{v}(s;w))^2]
$$


然后求梯度，更新参数：


$$
\begin{gather}
\Delta w=-\frac{1}{2}\alpha \nabla_wJ(w) \\
w_{t+1}=w_t+\Delta w
\end{gather}
$$


但是实际上我们是不知道真实值的，所以就利用之前model-free的方法去估计value函数，接下来就以$Q(s,a)$来说明了。


$$
\begin{gather}
J(w)=E_{\pi}[\frac{1}{2}(Q^{\pi}(s,a)-\hat{Q}(s,a;w))^2] \\
-\Delta J(w)=(Q^{\pi}(s,a)-\hat{Q}(s,a;w))\nabla_w\hat{Q}(s,a;w) \\
\end{gather}
$$



> 上式中的负号("-")在接下来的记号里为了方便就省略了。

对于不同的方法，用来替代$Q^{\pi}(s,a)$的表达式是不同的，

对于MC：


$$
\Delta J(w)=(G_t-\hat{Q}(s_t,a_t;w))\nabla_w\hat{Q}(s_t,a_t;w)
$$


对于Sarsa：


$$
\Delta J(w)=(R_{t+1}+\gamma\hat{Q}(s_{t+1},a_{t+1};w)-\hat{Q}(s_t,a_t;w))\nabla_w\hat{Q}(s_t,a_t;w)
$$


对于Q-learning:


$$
\Delta J(w)=(R_{t+1}+\gamma \underset{a}{\max}\hat{Q}(s_{t+1},a;w)-\hat{Q}(s_t,a_t;w))\nabla_w\hat{Q}(s_t,a_t;w)
$$


这里有一点需要注意一下，就是对于使用TD方法的Sarsa和Q-learning来说，从上面的式子可以看到，它们的target中也是包含待更新的参数的，所以其实上面的式子并不是一个梯度，这也是导致这些算法在训练的过程中很难收敛的原因之一，另外对于off-policy的Q-learning来说，还会面临用来训练的数据（基于behavior policy采样）的分布和要优化的策略（target policy）对应的数据的分布不一致的问题（这在监督学习的框架下是一个很大的问题），同样导致很难收敛。

因此关于RL的训练，有一个所谓的“死亡三角”（The Deadly Triad for the Danger of Instability and Divergence）

- **Function Approximation**：函数的表示形式的选择不当或者本身的表示能力不足
- **Bootstrapping**：主要是只TD方法引入的估计变差
- **Off-policy training**：训练数据和target policy产生的数据分布不一致

下面是一些关于收敛性的总结：

![28](https://pic.downk.cc/item/5f4139f0160a154a67d0bb34.png)

## Deep Q-Network

如果是使用Neural Network来做函数近似，既是非线性又是非凸的函数，要收敛就极其的困难了。

DeepMind 2015年在Nature上发了一篇强化学习的文章[Human-level control through deep reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)，使得深度强化学习正式走入大众的视野，文章就是利用neural network来近似Q函数，使用Q-learning，在很多Atari游戏中到达了精英级别的人类的水平甚至超过了人类。

下图是一个Atari游戏---Breakout的DQN图示:

![29](https://pic.downk.cc/item/5f4139f0160a154a67d0bb36.png)

由于强化学习的数据采集机制和TD learning性质，会给DQN的训练带来两个主要的问题：

1. 样本之间的高相关性
2. 不稳定的target

文章通过两个办法去缓解这两个问题：

1. Experience Replay
2. Fixed Q targets

### Experience Replay

它的思想就是准备一个replay memory $D$，把之前采集到的轨迹拆成一个一个的transition tuple$(s_t,a_t,r_t,s_{t+1})$，要更新梯度的时候从里面随机抽一个batch，从而降低样本之间的相关性。具体的做法，这里以一个transition tuple说明，batch update的话就是把batch里的所有样本的梯度求一个平均。

重复：

1. 从replay memory $D$中采样一个tuple $(s,a,r,s’)$

2. 计算target value：$y = r + \gamma \underset{a'}{\max}\hat{Q}(s',a';w)$

3. 随机梯度下降更新参数：$\Delta J(w)=(r+\gamma \underset{a'}{\max}\hat{Q}(s',a';w)-\hat{Q}(s,a;w))\nabla_w\hat{Q}(s,a;w)$

### Fixed Target

从上面的梯度更新式子可以看出，target value在每次更新之后都会发生变化，相当于你在估计一个不断变化的量，训练自然就非常困难，就像下图的猫捉老鼠一样，老鼠在动，猫要抓住它就很困难，那要是把老鼠固定住，那就容易多了。

![30](https://pic.downk.cc/item/5f4139f0160a154a67d0bb38.png)

所以Fixed Target的做法就是用另一套参数$w^-$作为target value的参数，而$w$用来训练，虽说是两套参数，但其实$w^-$是$w$的一段时间的滞后拷贝版。什么意思呢，最开始的时候，$w^-=w$，然后把$w^-$固定一定的training steps，用数据去更新$w$，更新完之后再把$w$的值赋值给$w^-$，如此重复训练，总结起来就是：

初始化：$w^-=w$

重复：

- for some training steps：

1. ​    从replay memory $D$中采样一个tuple $(s,a,r,s’)$
2. ​    计算target value：$y = r + \gamma \underset{a'}{\max}\hat{Q}(s',a';w^-)$
3. ​    随机梯度下降更新参数：$\Delta J(w)=(r+\gamma \underset{a'}{\max}\hat{Q}(s',a';w^-)-\hat{Q}(s,a;w))\nabla_w\hat{Q}(s,a;w)$

- $w^-=w$

### DQN代码

这次玩的是Atari游戏中的Pong，它其实有很多版本，不同版本的区别主要体现在是不是在连续几帧中使用同一个动作（Frame skip）以及是不是有一定的概率在下一次决策的时候沿用上一次的动作（Repeat action probability，不使用agent提供的动作），具体可以参考这篇博客[《强化学习环境学习-gym[atari]-paper中的相关设置](https://blog.csdn.net/qq_27008079/article/details/100126060)，如果觉得没说清楚可以再去官网[endtoend.ai](https://www.endtoend.ai/envs/gym/atari/)看一下原始说明，如果不想关注细节的话就直接忽略。在这里使用的是`PongNoFrameskip-v4`，它代表的意思是每一帧的动作都不同的，而且下一个动作不会去重复上一次的动作，完全听从agent的指派。再提一句，这个游戏的动作数是6，但其实只有上、下和不动3种动作，这是由于这款atari游戏环境中设定当action为0和1的时候球拍不动，为2和4的时候球拍向上运动，为3和5的时候向下运动。

![31](https://pic.downk.cc/item/5f4139f0160a154a67d0bb3b.png)

#### PyTorch版

```python
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

# wrapper

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height, 为了符合Pytorch处理图像的shape
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape  # channel在最后
        
        # 把channel放到最前面
        
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)/255.0  # 交换第1和第3位的位置, 并0-1化
    
def wrap_pytorch(env):
    return ImageToPyTorch(env)


name = "PongNoFrameskip-v4"
'''
下面三行是用来处理环境生成我们需要的输入, 需要用到baselines库里的几个函数, 当然也可以自己写, 
但是反正我现在是写不出来, 实在要写也行，只是太耗费时间了，等实际需要改环境的时候再研究吧，
可以直接用, 实在不放心就打印一下env.reset()的输出看一下是什么样的数据
'''
env = make_atari(name)
env = wrap_deepmind(env, frame_stack=False)
env = wrap_pytorch(env)


## Replay Memory

class ExperienceReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
   
    def __len__(self):
        return len(self.memory)
      
      
## Network

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=32,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        
    def forward(self, x):     # x should be torch.float type
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def feature_size(self):        
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).shape[1]
      

## Hyperparameters
class Config(object):
    def __init__(self):
        pass
    
config = Config()
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#epsilon variables    espislon-greedy exploration

config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
# epsilon随着epoch增加递减, 探索的越来越少

config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * np.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables

config.gamma = 0.99
config.lr = 1e-4

# memory

config.target_net_update_freq = 1000 # 每个多少的epoch更新target网络

config.exp_replay_size = 100000      # replay memory的容量

config.batch_size = 32

# learning control variables

config.learn_start = 10000     # 在replay memory中预先放入多少个tuple再开始训练

config.max_frames = 1000000    # 训练多少个steps



## Agent

class Agent(object):
    def __init__(self, static_policy=False, env=None, config=None):
        self.device = config.device
        
        self.gamma = config.gamma
        self.lr = config.lr
        self.target_net_update_freq = config.target_net_update_freq
        self.experience_replay_size = config.exp_replay_size
        self.batch_size = config.batch_size
        self.learn_start = config.learn_start
        
        self.static_policy = static_policy # 是train模式还evaluate模式
        
        self.env = env
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
            
        
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
                
        
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)        
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        
        # move to correct device
        
        self.model =self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        
            
        self.update_count = 0
        self.losses = []
        self.rewards = []
        
        
    def append_to_replay(self, s, a, r, s_, done):
            self.memory.push((s, a, r, s_, done))
                
            
    def pre_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        # print(batch_state.dtype)
        
        shape = (-1, ) + self.num_feats
        
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float).view(shape)
        batch_done = torch.tensor(batch_done, device=self.device, dtype=torch.int32).squeeze().view(-1, 1)  # 开始这里写成了batch_done, 导致loss不收敛。。。

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done


    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = batch_vars
        
        # estimate
        
        current_q_values = self.model(batch_state).ga
        ther(axis=1, index=batch_action)
        
        # target
        
        with torch.no_grad():
            # max_next_action = self.target_model(batch_next_state).max(dim=1)[1].view(-1, 1)
            
            max_next_q_values = self.target_model(batch_next_state).max(dim=1)[0].view(-1, 1)
            
            target = batch_reward + self.gamma * max_next_q_values * (1 - batch_done)
            
        diff = target - current_q_values
        loss = self.huber(diff)
        loss = loss.mean()
        
        return loss
    
    
    def update(self, s, a, r, s_, done, frame=0):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_, done)
        
        if frame < self.learn_start:
            return None
        
        batch_vars = self.pre_minibatch()
        
        loss = self.compute_loss(batch_vars)
        
        
        # optimize the model
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()
        
        # update target model
        
        self.update_count += 1
        if self.update_count % self.target_net_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())        
        
        self.losses.append(loss.item())
    
        
    def get_action(self, s, eps=0.1):   # epsilon-greedy policy
        
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                # print(s.dtype)
                
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(dim=1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)
            
               
    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)  


    def save_w(self):
        # Returns a dictionary containing a whole state of the module.
        
        torch.save(self.model.state_dict(), './model.dump')
        torch.save(self.optimizer.state_dict(), './optim.dump')
        
    def load_w(self):
        fname_model = './model.dump'
        fname_optim = './optim.dump'
        
        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())
            
        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))
            
    def save_replay(self):
        pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
        
    def load_replay(self):
        fname = './exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))    


            
## 可视化

def visualize(frame_idx, rewards, losses, elapsed_time):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    if rewards:
        plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
        plt.plot(rewards)
    if losses:
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
    plt.show()
            
     
    
## training loop

start = timer()

dqn_agent = Agent(env=env, config=config)
episode_reward = 0

observation = env.reset()

for frame_idx in range(1, config.max_frames+1):
    epsilon = config.epsilon_by_frame(frame_idx)
    
    action = dqn_agent.get_action(observation, epsilon)
    pre_observation = observation
    observation, reward, done, _ = env.step(action)
    
    dqn_agent.update(pre_observation, action, reward, observation, done, frame_idx)   # 之前这里漏了action, 导致loss怎么也输不出来, 搞了我一天。。。难受
    
    episode_reward += reward
    
    if done:
        observation = env.reset()
        dqn_agent.rewards.append(episode_reward)
        episode_reward = 0
        
        if np.mean(dqn_agent.rewards[-10:]) > 19:
            visualize(frame_idx, dqn_agent.rewards, dqn_agent.losses, timedelta(seconds=int(timer()-start)))
            break

    if frame_idx % 10000 == 0:
        visualize(frame_idx, dqn_agent.rewards, dqn_agent.losses, timedelta(seconds=int(timer()-start)))
        
dqn_agent.save_w()

env.close()    

```

#### TensorFlow版

```python
'''
但其实一个强化学习的框架最好应该分成4部分：
1. ReplayMemory, 存数据, 整理数据, 取数据
2. Network, 确定网络的框架, 比如MLP, CNN, RNN等
3. Algorithm, 是用什么RL方法, DQN, PPO等
4. Agent, 负责与环境进行交互, 这样对于不同的问题前面三个部分可以做到最大程度的共享, 但也是需要专门设计一下前面三个的架构才能做到, 也不是那么容易直接拿来用的
'''


import numpy as np
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.list_physical_devices(device_type='GPU')

import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

'''
TensorFlow就不需要将图片格式进行转化了, 默认的处理形式: [width, height, channel]
'''

env = make_atari(name)
env = wrap_deepmind(env, frame_stack=False, scale=True)


## 将收集数据、提取数据和整理数据全部整合到replay memory里

class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = []
        self.maxsize = size
        self.next_idx = 0
        
    def __len__(self):
        return len(self.storage)
    
    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage)-1) for _ in range(batch_size)]  # 注意这里random.randint是包括两个端点的，所以右端点要-1，不然可能会报错out of range
        
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        data = self.storage[0]
        ob_dtype = data[0].dtype
        ac_dtype = data[1].dtype 
        for i in idxes:
            data = self.storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t, dtype=ob_dtype), np.array(actions, dtype=ac_dtype), np.array(rewards, dtype=np.float32), np.array(obses_tp1, dtype=ob_dtype), np.array(dones, dtype=np.float32)



## Network

class DQNNetwork(keras.Model):
    def __init__(self, action_dim):
        super(DQNNetwork, self).__init__()
        
        self.action_dim = action_dim
        
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(self.action_dim)
        
    def call(self, obs):
        x = tf.cast(obs, dtype=tf.float32)
        x = tf.nn.relu(self.conv1(obs))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))  # keras.activations.relu(self.conv3(x))
        
        x = self.flatten(x)
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)

        return x


## test ====================================================

# num_feats, num_actions = env.observation_space.shape, env.action_space.n

# env.reset()

# obs, _,_, _  = env.step(1)

# obs = tf.expand_dims(tf.constant(obs, dtype=tf.float32), axis=0)

# model = DQNNetwork(num_actions)

# print(model(obs).shape)

#=============================================================================


## Hyperparameters

class Config(object):
    def __init__(self):
        pass
    
config = Config()
#epsilon variables    espislon-greedy exploration

config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
# epsilon随着epoch增加递减, 探索的越来越少

config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * np.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables

config.gamma = 0.99
config.lr = 1e-4

# memory

config.target_net_update_freq = 1000 # 每个多少的epoch更新target网络

config.exp_replay_size = 100000      # replay memory的容量

config.batch_size = 32

# learning control variables

config.learn_start = 10000     # 在replay memory中预先放入多少个tuple再开始训练

config.max_frames = 1000000    # 训练多少个steps


## Algorithm，我发现不是很好把这一块从agent中拆出来，还是先放着吧。。。

class DQNAlgorithm(object):
    def __init__(self):
        pass
    


## Agent

class DQNAgent(object):
    def __init__(self, network=DQNNetwork, eval_mode=False, env=None, config=None):
        
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        self.gamma = config.gamma
        self.lr = config.lr
        self.target_net_update_freq = config.target_net_update_freq
        self.experience_replay_size = config.exp_replay_size
        self.batch_size = config.batch_size
        self.learn_start = config.learn_start
        
        self.network = network
        self.eval_mode = eval_mode
        
        self.memory = ReplayBuffer(self.experience_replay_size)
        
        self.model = self.network(self.action_dim)
        self.target_model = self.network(self.action_dim)
        # self.model.build(input_shape=[None]+list(self.obs_dim))
        
        # self.target_model.build(input_shape=[None]+list(self.obs_dim))
        
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = optimizers.Adam(lr=self.lr)
        self.loss = tf.losses.Huber(delta=1.)
        
        self.update_count = 0
        self.losses = []
        self.rewards = []
        
    
    def train(self, frame=0):
        if self.eval_mode:
            return None
        
        if frame < self.learn_start:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        obses_t, actions, rewards, obses_tp1, dones = transitions
        
        
        with tf.GradientTape() as tape:
            # Q(s,a)
            q_vals = self.model(obses_t)
            indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
            chosen_q_vals = tf.gather_nd(q_vals, indices=indices)
            # print('q_vals shape:{}, chosen_q_vals shape:{}'.format(q_vals.shape, chosen_q_vals.shape))
            
            q_tp1_vals = tf.math.reduce_max(self.target_model(obses_tp1), axis=-1)
            
            targets = tf.stop_gradient(rewards + self.gamma * q_tp1_vals * (1-dones))
            
            loss = self.loss(chosen_q_vals, targets)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1, 1) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.losses.append(loss.numpy())
        
        # update target model
        self.update_count += 1
        if self.update_count % self.target_net_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
            
    
    def get_action(self, obs, eps=0.1):
        if np.random.random() >= eps or self.eval_mode:
            obs = np.expand_dims(obs, 0)
            a = np.argmax(self.model(obs), axis=-1)
            return a
        else:
            return np.array(np.random.randint(0, self.action_dim))

    def huber_loss(self, x, delta):
        x = tf.cast(x, dtype=tf.float32)
        return tf.where(tf.abs(x) < delta, 0.5 * x**2, delta * (tf.abs(x) - 0.5 * delta))


    def save_w(self):
        self.model.save_weights('./model_weights.ckpt')


    def load_w(self):
        fname_model = './model_weights.ckpt'
        
        if os.path.isfile(fname_model):
            self.model.load_weights(fname_model)

    def save_replay(self):
        pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
        
    def load_replay(self):
        fname = './exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))    




## 可视化
def visualize(frame_idx, rewards, losses, elapsed_time):
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    if rewards:
        plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
        plt.plot(rewards)
    if losses:
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
    plt.show()



## training loop

start = timer()

dqn_agent = DQNAgent(env=env, config=config)
episode_reward = 0

observation = env.reset()

for frame_idx in range(1, config.max_frames+1):
    epsilon = config.epsilon_by_frame(frame_idx)
    
    action = dqn_agent.get_action(observation, epsilon)
    pre_observation = observation
    observation, reward, done, _ = env.step(action)
    
    dqn_agent.memory.add(pre_observation, action, reward, observation, done)
    # print(len(dqn_agent.memory.storage))
    
    dqn_agent.train(frame_idx)
    episode_reward += reward
    
    if done:
        observation = env.reset()
        dqn_agent.rewards.append(episode_reward)
        episode_reward = 0
        
        if np.mean(dqn_agent.rewards[-10:]) > 19:
            visualize(frame_idx, dqn_agent.rewards, dqn_agent.losses, timedelta(seconds=int(timer()-start)))
            break

    if frame_idx % 10000 == 0:
        visualize(frame_idx, dqn_agent.rewards, dqn_agent.losses, timedelta(seconds=int(timer()-start)))
        
dqn_agent.save_w()

env.close()

```

GitHub上的也有一些比较好的RL代码库，比如[spinningup](https://github.com/openai/spinningup)，[baselines](https://github.com/openai/baselines)，[garage](https://github.com/rlworkgroup/garage)，[rllab](https://github.com/ray-project/ray/blob/master/python/ray/rllib)，但是它们都是有一个代码架构的（确实是应该有，但是其实对初学者不是很友好），基本上涉及到某一个具体的算法，它的一些核心组件都是散落在不同的文件夹里，为了看懂一个算法，可能要索引很多层文件夹才能把所有代码都搜索到，有时候看着看着自己就晕了（我自己就是这样），所以目前我就先把所有的代码放到一起，可能到后面慢慢的我也会采用架构的形式来写了。

下一次就写一下各种DQN的Variants。