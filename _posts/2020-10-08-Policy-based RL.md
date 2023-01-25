---
layout:     post
title:      Reinforcement Learning 6
subtitle:   Policy-based Reinforcement Learning
date:       2020-10-08
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

这一篇我们来介绍policy-based RL，之前讲的都是基于value函数来导出策略，我们也可以直接将策略函数参数化$\pi_{\theta}(a\|s)$，然后通过优化方法来得到最优策略。

那policy-based RL和value-based RL比有什么优劣势呢，总结一下（也不是我总结的，从周老师的课件里搬运过来）：

**优势**

- 更好的收敛性保证（能收敛，但是不是全局最优就不好说了）
- 对于高维或者连续的动作空间更有效率
- 可以学习stochastic策略，value-based RL不行（后面通过改进也是可以学习的）

**劣势**

- 通常是收敛到局部最优
- 在评估策略的时候方差会很大（这个是一个结果性的说明，就是说在评估策略的时候，比如论文里展示用学习到的策略去跑游戏得到的returns，会看到波动很大，可能和它本身是随机性策略有关）

## 常见的参数化策略方法

### Softmax policy for discrete action space

$$
\pi_{\theta}(a|s)=\frac{\exp f_{\theta}(s,a)}{\sum_{a'}\exp f_{\theta}(s,a')}
$$

### Gaussian policy for continuous action space

均值是状态特征的一个函数：$\mu(s)=f_{\theta}(s)$

方差可以是给定的$\sigma^2$，也可以被参数化

然后$a\sim N(\mu(s), \sigma^2)$

## Likelihood ratio trick

这个技巧在后面推导policy gradient的时候会用到，所以这里先介绍一下，其实就是：

$$
\nabla_{\theta}\pi_{\theta}(a|s)=\pi_{\theta}(a|s)\frac{\nabla_{\theta}\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}=\pi_{\theta}(a|s)\nabla_{\theta}\log\pi_{\theta}(a|s)
$$

## 目标函数

对于policy-based RL来说，它要优化的目标是：$J(\theta)=E_{\tau \sim \pi_{\theta}}[\sum_t\gamma^tR(s_t^{\tau}, a_t^{\tau})]$

其中$\tau$是一条采用策略$\pi_{\theta}$从环境中采样出来的轨迹（假设有terminal state）：$\tau = (s_0, a_0, r_1, \ldots,s_{T-1}, a_{T-1}, r_T,s_T) \sim (\pi_{\theta}, P(s_{t+1}\|s_t, a_t))$

所以我们的目标就是：$\theta^*=\arg \underset{\theta}{\max}J(\theta)$

我们把$J(\theta)$完全写成关于$\tau$的函数：$J（\theta)=E_{\tau}[G_{\tau}]=\sum_{\tau}P(\tau;\theta)G(\tau)$

其中$P(\tau;\theta)=\mu(s_0)\prod_{t=0}^{T-1}\pi_{\theta}(a_t\|s_t)P(s_{t+1}\|s_t,a_t)$，表示当执行策略$\pi_{\theta}$时，$\tau$出现的概率，这里用到了markov property（但是实际上后面算policy gradient的时候会发现，其实并不需要markov property）。对所有的$\tau$进行求和在实际操作中可以用$m$条轨迹来近似。$G_{\tau}$就和之前介绍的return的含义是一样的（discounted sum of rewards），只是现在是从轨迹的角度来看。

然后我们就可以对$J(\theta)$进行求导了：

$$
\nabla_{\theta} J(\theta)=\sum_{\tau}\nabla_{\theta}P(\tau;\theta)G(\tau)=\sum_{\tau}P(\tau;\theta)G(\tau)\nabla_{\theta} \log P(\tau;\theta)
$$

我们再来看$\nabla_{\theta} \log P(\tau;\theta)$这一部分等于什么：

$$
\nabla_{\theta} \log P(\tau;\theta)=\nabla_{\theta} \log [\mu(s_0)\prod_{t=0}^{T-1}\pi_{\theta}(a_t|s_t)P(s_{t+1}|s_t,a_t)]=\sum_{t=0}^{T-1}\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)
$$

可以发现那些转移概率的部分因为和$\theta$无关，全部都消去了，最后就剩下策略函数的score function求和了（如果把$\pi_{\theta}$看成似然函数的话），所以**policy gradient本身是model-free的**，不需要知道dynamics model。

所以把上面的公式整理一下我们就得到policy gradient：

$$
\nabla_{\theta} J(\theta)=\sum_{\tau}P(\tau;\theta)G(\tau)\sum_{t=0}^{T-1}\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\approx\frac{1}{m}\sum_{i=1}^mG(\tau_i)\sum_{t=0}^{T-1}\nabla_{\theta}\log \pi_{\theta}(a_t^i|s_t^i)
$$

## Problem of policy gradient

由于$G(\tau)=\sum_t\gamma^tR(s_t,a_t)$，采样$m$条轨迹的问题实际上就是MC的问题：虽然是无偏的，但是方差很大。

先放一张图，后面再解释

![41](https://pic.downk.cc/item/5f7eb9fa1cd1bbb86badb56f.png)

有3种方法可以缓解这个问题：

- use temporal causality
- include a baseline
- using critic

## Reducing variance of policy gradient using causality

回顾一下policy gradient的公式：$\nabla_{\theta}J(\theta)=\nabla_{\theta}E_{\tau}[G(\tau)]=E_{\tau}[(\sum_{t=0}^{T-1}\gamma^tr_t)(\sum_{t=0}^{T-1}\nabla_{\theta}\log \pi_{\theta}(a_t^i\|s_t^i))]$

所谓的时序因果性，其实可以概括成两句话（它们其实是等价的，只是说法不同）：

- 对于某一时刻的$r_t$，它应该只和当前$t$以及之前$t$的score function有关
- 对于某一时刻$t$的策略，它仅影响当前时刻以及之后时刻的reward

英文是：policy at time $t'$ can not affect reward at time $t$ when $t<t'$

于是，我们可以把policy gradient中的一些项去掉，变成：

$$
\nabla_{\theta}E_{\tau}[G(\tau)]=E_{\tau}[\sum_{t=0}^{T-1}(\sum_{t'=t}^{T-1}\gamma^tr_t)\nabla_{\theta}\log \pi_{\theta}(a_t^i|s_t^i)]=E_{\tau}[\sum_{t=0}^{T-1}G_t\nabla_{\theta}\log \pi_{\theta}(a_t^i|s_t^i)]
$$

注意下标的变化，$G_t$就是我们之前value-based RL里的那个从$t$时刻开始之后的reward的折现和，我们一般称其为**reward to go**。去掉了一些具有随机性的$r_t$，方差自然就小了。

### REINFORCE

[Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

Williams 1992年提出了这个MC policy gradient的算法，就是考虑了时序因果的policy gradient，算法的流程如下图：

![40](https://pic.downk.cc/item/5f7eba7f1cd1bbb86baddb92.png)

#### Pytorch codes

```python
import argparse
import gym
import numpy as np
import pandas as pd
from itertools import count
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# hyparameter

# seed = 543

gamma = 0.99
is_render = False

# env = gym.make('CartPole-v1')    

# 我还没搞清楚v0和v1的区别（搞清楚了，是它们的游戏分值上限不同，v0是195，v1是475）

# env.seed(seed)

# torch.manual_seed(seed)


class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Policy, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        self.affine1 = nn.Linear(self.input_shape, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, self.action_dim)



    def forward(self, obs):   # obs must be a tensor

        x = self.affine1(obs)
        x = self.dropout(x)
        x = F.relu(x)
        action_logits = self.affine2(x)
        actions = F.softmax(action_logits, dim=-1)
        return actions



class REINFORCEAgent(object):
    def __init__(self, env_name=None, policy=Policy, eval_mode=False):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.policy = policy(self.obs_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)


        self.eval_mode = eval_mode

        if self.eval_mode:
            self.policy.eval()
        else:
            self.policy.train()

        self.log_probs = []    # 用来记录每个时刻t的log(pi(a_t|s_t))

        self.rewards = []            # 用来记录每个时刻t的reward, r_t

        self.returns = []            # 用来记录每个时刻t的return, G_t

        self.loss = []               # 用来记录每个时刻t的loss: G_t * log(pi(a_t|s_t))

        self.eps = np.finfo(np.float32).eps.item()     # 创建一个很小的浮点数，加在分母，防止0的出现


    def select_action(self, obs):  # obs is not a tensor

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0)   # [1, obs_dim]

        probs = self.policy(obs)   # 产生策略函数，是一个关于action的概率

        m = Categorical(probs)     # 生成一个Categorical分布，在CartPole里是二项分布

        action = m.sample()        # 从分布里采样，采出的是索引

        self.log_probs.append(m.log_prob(action))  # 把对应的log概率记录下来, 因为后面导数是对logπ（θ）来求的

        return action.item()


    def train(self):
        R = 0
        # policy gradient update

        for r in self.rewards[::-1]:       # 倒序

            R = r + gamma * R              # 计算t到T的reward折现和

            self.returns.insert(0, R)      # 在最前面插入


        returns = torch.tensor(self.returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)   # 把returns做一个标准化，这样对于action的矫正会更有效果一些，因为一条成功的轨迹里并不一定是所有action都是好的


        for log_prob, R in zip(self.log_probs, returns):
            self.loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(self.loss).sum()  # self.loss 是一个列表，里面元素是tensor，然后cat一下, 为了能反向传播梯度？？？

        '''    
        这个loss的计算有些trick，我一开始是这么写的
        returns = self.returns
        ...
        loss = torch.tensor(self.loss, requires_grad=True).sum()
        结果return就训不上去，我还没搞明白原因 
        '''

        loss.backward()
        self.optimizer.step()

        del self.rewards[:]                        # 把列表清空，但是列表还在，[]

        del self.returns[:]
        del self.log_probs[:]
        del self.loss[:]


    def eval_(self, n_trajs=5):
        self.policy.eval()
        env = gym.make(self.env_name)
        returns = []
        for i in range(n_trajs):
            ep_return = 0
            obs = env.reset()
            for step in range(1000):
                action = self.select_action(obs)
                obs, reward, done, _ = env.step(action)
                ep_return += reward

                if done:
                    returns.append(ep_return)
                    break
        env.close() 
        self.policy.train()
        return np.array(returns).mean()


    def render_(self):
        self.policy.eval()
        env = gym.make(self.env_name)
        obs = env.reset()
        for _ in range(1000):
            env.render()
            action = self.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.close()
        self.policy.train()


# training Loop

start = timer()
running_returns = []
agent_reinforce = REINFORCEAgent(env_name='CartPole-v1', policy=Policy)

for episode in count(1): # 一直加1的while, 表示一条episode

    # print('episode%d'%episode)

    obs, ep_return = agent_reinforce.env.reset(), 0
    for step in range(1000):
        action = agent_reinforce.select_action(obs)
        obs, reward, done, _ = agent_reinforce.env.step(action)
        if is_render:
            agent_reinforce.render()
        agent_reinforce.rewards.append(reward)
        ep_return += reward
        if done:
            running_returns.append(ep_return)
            break

    agent_reinforce.train()


    if episode % 10 == 0:
        # clear_output(True)

    plt.plot(pd.Series(running_returns).rolling(100, 20).mean())
        plt.title('episide:{}, time:{},'.format(episode, timedelta(seconds=int(timer()-start))))
        plt.show()
    if np.array(running_returns)[-10:].mean() > agent_reinforce.env.spec.reward_threshold:
        eval_return = agent_reinforce.eval_()
        print("Solved! eval return is now {}!".format(eval_return))
        break 
```

## Reducing Variance using Baseline

第二种方法是给$G_t$减掉一个baseline $b(s_t)$，把policy gradient变成：

$$
\nabla_{\theta}E_{\tau}[G(\tau)]=E_{\tau}[(\sum_{t=0}^{T-1}(G_t-b(s_t))\nabla_{\theta}\log \pi_{\theta}(a_t^i\|s_t^i)]
$$

这就有点像，你有独立同分布的$n$个数据$x_1, x_2, \ldots, x_n$，$Var(x_1 - \bar{x})<Var(x_1)$。

那加什么呢，a good baseline is the expected return：$b(s_t)=E[G_t]=E[r_t+r_{t+1}+\ldots+r_{T-1}]$，首先它能使方差减小，另外还有一个好处，就是加了之后不改变原来的期望：

$$
E_{\tau}[\nabla_{\theta}\log\pi_{\theta}(\tau)b]=\int \pi_{\theta}(\tau)\nabla_{\theta}\log\pi_{\theta}(\tau)bd\tau=\int \nabla_{\theta}\pi_{\theta}(\tau)bd\tau=b \nabla_{\theta}\int \pi_{\theta}(\tau)d\tau=0
$$

但是这并不是最优的baseline，最优的是什么呢，这里参考了[UC Berkeley CS285 深度学习课程](http://rail.eecs.berkeley.edu/deeprlcourse/)，直接放图，省去打公式的麻烦。

![42](https://pic.downk.cc/item/5f7ebc2e1cd1bbb86bae567b.png)

进一步的，我们对$b(s_t)$也进行参数化，即$b_w(s_t)$，通过学习参数来得到好的baseline，这就是Vanilla Policy Gradient Algorithm with Baseline，[Sutton, McAllester, Singh, Mansour (1999). Policy gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

### Vanilla Policy Gradient Algorithm with Baseline

![43](https://pic.downk.cc/item/5f7ebc6f1cd1bbb86bae6482.png)

#### Pytorch codes

```python
# ==============Vanilla Policy Gradient Algorithm with Baseline   ---Pong=========================

'''
我算是搞明白Pong这个游戏的门道了，就属它事情多。。。
它的一盘游戏是以玩家双方某一方谁到达20分就结束，此时它的done才会变成True，其他状态都是False，
这就意味着它一盘里是有很多局的，而一局其实才是我们在一般游戏里说的玩一次采样的轨迹，
那怎么判断是不是完成了一局呢，Pong has either +1 or -1 reward exactly when game ends
就是说，在一局结束时，如果你赢了，reward是1，你输了，reward是-1，其他情况reward都是0，
所以是通过reward是否非0来判断一局结束与否

再吐槽一下，明明游戏里的动作只有3种，不动，向上、向下，却非要每种动作用2个数字来表示，搞出6维的动作，何必呢。。。
'''

# import argparse
import gym
import numpy as np
import pandas as pd
from itertools import count
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
from IPython.display import clear_output
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# is_cuda = torch.cuda.is_available()

gamma = 0.99
decay_rate = 0.99 # decay rate for RMSprop, weight decay

learning_rate = 3e-4
batch_size = 20
seed = 87
test = False # whether to test the trained model or keep training

render = True if test else False


D = 80 * 80 # 输入的维度，后面会对状态进行预处理，裁剪，下采样，去除背景，差分

def preprocess(I):
    """ 
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector 
    """
    I = I[35:195]        # 裁剪

    I = I[::2, ::2, 0]   # 下采样，缩小一半

    I[I==144] = 0        # 擦除背景（background type 1）

    I[I==109] = 0        # 擦除背景（background type 2）

    I[I!=0] = 1          # 转为灰度图

    return I.astype(np.float).ravel()


PGbaselineType = namedtuple('pgbaseline', ['action_prob', 'baseline'])

class PGbaselineNetwork(nn.Module):
    def __init__(self, input_shape=None, action_dim=None):
        super(PGbaselineNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.affine1 = nn.Linear(self.input_shape, 256)                # 两个部分共用一个特征提取网络

        self.action_head = nn.Linear(256, self.action_dim)   # 策略函数

        self.baseline = nn.Linear(256, 1)               # baseline函数



    def forward(self, obs):   # obs must be a tensor

        x = F.relu(self.affine1(obs))

        action_logits = self.action_head(x)
        baseline = self.baseline(x)

        # return PGbaselineType(F.softmax(action_logits, dim=-1), baseline)        

        return F.softmax(action_logits, dim=-1), baseline




class PGbaselineAgent(object):
    def __init__(self, env_name=None, obs_dim=None, action_dim=None, net=PGbaselineNetwork, eval_mode=False):

        self.env_name = env_name
        self.env = gym.make(self.env_name)


        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = net(self.obs_dim, self.action_dim)
        # self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate, weight_decay=decay_rate)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        '''
        优化算法和学习率的影响好大啊，用RMSprop算了半个小时，没跑通，用Adam 5分钟跑通了CartPole...
        但是换成Pong这个游戏，Adam就又不行了，训了一晚上也不见起色(也不一定，只训了5个小时...)

        这里其实有一个问题，就是我是以每一局作为一条轨迹来计算return的，也就是得1分或者输1分，
        但是不是应该以一盘作为一条轨迹，agent可能赢了几次，也输了几次，然后以这一条长的轨迹来计算discounted cumulative reward
        '''

        if eval_mode:
            self.net.eval()
        else:
            self.net.train()


        self.log_probs_baseline = []
        self.rewards = []
        self.returns = []
        self.loss = []
        self.baseline_loss = []


    def select_action(self, obs):  # obs is not a tensor

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0)
        probs, baseline = self.net(obs)
        m = Categorical(probs)
        action = m.sample()

        self.log_probs_baseline.append((m.log_prob(action), baseline))

        return action.item()


    def compute_return(self):
        R = 0
        for r in self.rewards[::-1]:
            R = r + gamma * R
            self.returns.insert(0, R)
        del self.rewards[:]    



    def train(self):

        returns = torch.tensor(self.returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for (log_prob, baseline), R in zip(self.log_probs_baseline, returns):
            advantage = R - baseline
            self.loss.append(-log_prob * advantage)                               # policy gradient
            self.baseline_loss.append(F.smooth_l1_loss(baseline.squeeze(), R))        # baseline function approximation

        self.optimizer.zero_grad()
        policy_loss = torch.stack(self.loss).sum()
        baseline_loss = torch.stack(self.baseline_loss).sum()
        loss = policy_loss + baseline_loss

        loss.backward()

        self.optimizer.step()

        print('loss: {:2f}---policy_loss: {:2f}---baseline_loss: {:2f}'.format(loss.item(), policy_loss.item(), baseline_loss.item()))

        del self.log_probs_baseline[:]
        del self.returns[:]
        del self.loss[:]
        del self.baseline_loss[:]


    def eval_(self, n_trajs=5):
        self.policy.eval()
        env = gym.make(self.env_name)
        returns = []
        for i in range(n_trajs):
            ep_return = 0
            obs = env.reset()
            for step in range(10000):
                action = self.select_action(obs)
                obs, reward, done, _ = env.step(action)
                ep_return += 1 if reward==0 else 0

                if done:
                    returns.append(ep_return)
                    break
        env.close()   
        self.policy.train()
        return np.array(returns).mean()



    def render_(self):
        self.net.eval()
        env = gym.make(self.env_name)
        obs = env.reset()
        for _ in range(10000):
            env.render()
            action = self.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.close()
        self.net.train()


    def save(self, step):
        torch.save(self.policy.state_dict(), './vanilla_{}.pkl'.format(step))

    def load(self, path):
        if os.path.isfile(path):
            self.policy.load_state_dict(torch.load(path))
            # self.policy.load_state_dict(torch.load(path), map_location=lambda storage, loc: storage))  # 在gpu上训练，load到cpu上的时候可能会用到

        else:
            print('No "{}" exits for loading'.format(path))



# training Loop

start = timer()
running_returns = []
agent_pgbaseline = PGbaselineAgent(env_name='Pong-v0', obs_dim=D, action_dim=2, net=PGbaselineNetwork)

for episode in count(1): # 一直加1的while, 表示一盘游戏，先得到20分胜

    # print('episode%d'%episode)

    obs, ep_return = agent_pgbaseline.env.reset(), 0
    prev_x = None
    for step in range(10000):
        curr_x = preprocess(obs)
        x = curr_x - prev_x if prev_x is not None else np.zeros(D)
        # 输入的其实是差分

        prev_x = curr_x
        action = agent_pgbaseline.select_action(x)
        action_env = action + 2    # 不考虑static动作，模型输出的是0，1， 环境中变成2，3，分别对应上、下

        obs, reward, done, _ = agent_pgbaseline.env.step(action_env)

        agent_pgbaseline.rewards.append(reward)
        ep_return += 1 if reward == 1 else 0

        '''
        这是Pong特有的
        '''

        if reward != 0:
            agent_pgbaseline.compute_return()   # 这一步可以放在done里试试

        if done:
            running_returns.append(ep_return)
            break

    if episode % 10 == 0:
        agent_pgbaseline.train()


    if episode % 20 == 0:
        # clear_output(True)

        plt.plot(pd.Series(running_returns).rolling(100, 20).mean())
        plt.title('episide:{}, time:{},'.format(episode, timedelta(seconds=int(timer()-start))))
        plt.show()
    if np.array(running_returns)[-10:].mean() > 10:    # 降低一点标准看看能不能跑通

        eval_return = agent_reinforce.eval_()
        print("Solved! eval return is now {}!".format(eval_return))
        break 

#======================================================================================================================
```

## Reducing variance using Critic

第三种减小方差的方式是使用Critic，什么是Critic呢？我们还是回到policy gradient里的$G_t$，它其实就是通过MC方法采样得到的，本质上是$Q^{\pi\theta}(s_t,a_t)$的一个无偏但是noisy的估计。那我们也可以不用MC的方法，换一种方法来估计$Q^{\pi\theta}(s_t,a_t)$，通过Critic来估计，$Q_w(s,a)\approx Q^{\pi\theta}(s,a)$，为什么要叫Critic呢，Critic可以翻译成评论家，与之对应的是Actor，在这里就是指策略函数，Actor负责来执行（表演），Critic负责来指导，知道的方法就是提供正确的$Q^{\pi\theta}(s,a)$，所以就有了**Actor-Critic Policy Gradient**

$$
\nabla_{\theta}J(\theta)=E_{\pi(\theta)}[\sum_{t=0}^{T-1}Q_w(s_t,a_t)\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)]
$$

整个网络有两套参数:

1. Actor：policy parameter $\theta$
2. Critic：value function parameter $w$

而关于value函数的估计，之前介绍过的value-based RL的方法就可以拿来用了，比如TD learning。

### Actor-Critic with baseline

AC也可以加baseline，而且对于Q函数，有一个绝佳的baseline，就是V函数，$V^{\pi}(s)=E_{a\sim\pi}[Q^{\pi}(s,a)]$。

类比之前的Vanilla PG 算法，也定义**Advantage fuction**：$A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$，所以policy gradient进一步变成（省略了对$t$求和）

$$
\nabla_{\theta}J(\theta)=E_{\pi(\theta)}[A^{\pi}(s,a)\nabla_{\theta}\log \pi_{\theta}(a|s)]
$$

我们通常会给它一个称谓：**Advantage Actor-Critic (A2C)**

还有一件事，就是A2C看上去似乎是需要对V和Q分别用一套网络参数去拟合，然后计算Advantage function，但是其实可以有更简单的方法，直接得到一个Advantage function的估计，就是利用TD error $\delta^{\pi_{\theta}}=R(s,a)+\gamma V^{\pi_{\theta}}(s')-V^{\pi_{\theta}}(s)$，因为它的期望就是Advantage function:

$$
E_{\pi_{\theta}}[\delta^{\pi_{\theta}}|s,a]=E_{\pi_{\theta}}[r+\gamma V^{\pi_{\theta}}(s')|s,a]-V^{\pi_{\theta}}(s)=Q^{\pi_{\theta}}(s,a)-V^{\pi_{\theta}}(s)=A^{\pi_{\theta}}(s,a)
$$

这样，我们就只需要估计V就能计算Advantage function了，然后就可以写代码了。

#### A2C Pytorch codes

```python
# =============A2C==========================================

import gym
import numpy as np
import pandas as pd
from itertools import count
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
from IPython.display import clear_output
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# is_cuda = torch.cuda.is_available()

gamma = 0.99
decay_rate = 0.99 # decay rate for RMSprop, weight decay

learning_rate = 3e-4
batch_size = 20
# seed = 87

test = False # whether to test the trained model or keep training

render = True if test else False

# env = gym.make('Pong-v0')

# env.seed(args.seed)

# torch.manual_seed(args.seed)


D = 80 * 80 # 输入的维度，后面会对状态进行预处理，裁剪，下采样，去除背景，差分

def preprocess(I):
    """ 
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector 
    """
    I = I[35:195]        # 裁剪

    I = I[::2, ::2, 0]   # 下采样，缩小一半

    I[I==144] = 0        # 擦除背景（background type 1）

    I[I==109] = 0        # 擦除背景（background type 2）

    I[I!=0] = 1          # 转为灰度图

    return I.astype(np.float).ravel() 



A2CType = namedtuple('a2c', ['action_prob', 'value'])

class A2CNetwork(nn.Module):
    def __init__(self, input_shape=None, action_dim=None):
        super(A2CNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.affine1 = nn.Linear(self.input_shape, 256)                # 两个部分共用一个特征提取网络

        self.action_head = nn.Linear(256, self.action_dim)   # 策略函数

        self.values = nn.Linear(256, 1)               # value函数



    def forward(self, obs):   # obs must be a tensor

        x = F.relu(self.affine1(obs))

        action_logits = self.action_head(x)
        values = self.values(x)

        # return A2CType(F.softmax(action_logits, dim=-1), values)        

        return F.softmax(action_logits, dim=-1), values



class A2CAgent(object):
    def __init__(self, env_name=None, obs_dim=None, action_dim=None, net=A2CNetwork, eval_mode=False):

        self.env_name = env_name
        self.env = gym.make(self.env_name)


        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = net(self.obs_dim, self.action_dim)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate, weight_decay=decay_rate)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        '''

        '''

        if eval_mode:
            self.net.eval()
        else:
            self.net.train()


        self.log_probs_values = []    # 里面的元素也是列表，每个列表表示一盘（局）游戏？

        self.rewards = []
        self.Q_values = []
        self.loss = []
        self.value_loss = []


    def select_action(self, obs):  # obs is not a tensor

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0)
        probs, values = self.net(obs)
        m = Categorical(probs)
        action = m.sample()

        self.log_probs_values[-1].append((m.log_prob(action), values))     # 注意和之前的不同

        return action.item()



    def train(self):
        R = 0
        for episode_id, episode_reward_list in enumerate(self.rewards):
            for i, r in enumerate(episode_reward_list):
                if i == len(episode_reward_list) - 1:
                    R = torch.scalar_tensor(r)
                else:     # self.log_probs_values.shape: [[(porbs, values), (porbs, values), (porbs, values)], [...], [...], [...]]

                    R = r + gamma * self.log_probs_values[episode_id][i+1][1]     # Q = r + gamma * V(s')

                self.Q_values.append(R)

        flatten_log_probs_values = [sample for episode in self.log_probs_values for sample in episode]



        for (log_prob, value), Q in zip(flatten_log_probs_values, self.Q_values):
            advantage = Q - value    # A(s,a) = Q(s,a) - V(s)，但其实这里已经是用了Advantage function的估计值 delta了，delta = r+gamma*V(s')-V(s)

            self.loss.append(-log_prob * advantage)                               # policy gradient

            self.value_loss.append(F.smooth_l1_loss(value.squeeze(), Q.squeeze()))        # value function approximation

        self.optimizer.zero_grad()
        policy_loss = torch.stack(self.loss).sum()
        value_loss = torch.stack(self.value_loss).sum()
        loss = policy_loss + value_loss

        loss.backward()

        self.optimizer.step()

        print('loss: {:2f}---policy_loss: {:2f}---value_loss: {:2f}'.format(loss.item(), policy_loss.item(), value_loss.item()))

        del self.log_probs_values[:]
        del self.rewards[:]
        del self.Q_values[:]
        del self.loss[:]
        del self.value_loss[:]


    def eval_(self, n_trajs=5):
        self.policy.eval()
        env = gym.make(self.env_name)
        returns = []
        for i in range(n_trajs):
            ep_return = 0
            obs = env.reset()
            for step in range(10000):
                action = self.select_action(obs)
                obs, reward, done, _ = env.step(action)
                ep_return += 1 if reward==0 else 0

                if done:
                    returns.append(ep_return)
                    break
        env.close()   
        self.policy.train()
        return np.array(returns).mean()



    def render_(self):
        self.net.eval()
        env = gym.make(self.env_name)
        obs = env.reset()
        for _ in range(10000):
            env.render()
            action = self.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.close()
        self.net.train()


    def save(self, step):
        torch.save(self.policy.state_dict(), './a2c_{}.pkl'.format(step))

    def load(self, path):
        if os.path.isfile(path):
            self.policy.load_state_dict(torch.load(path))
            # self.policy.load_state_dict(torch.load(path), map_location=lambda storage, loc: storage))  # 在gpu上训练，load到cpu上的时候可能会用到

        else:
            print('No "{}" exits for loading'.format(path))



# training Loop

start = timer()
running_returns = []
agent_a2c = A2CAgent(env_name='Pong-v0', obs_dim=D, action_dim=2, net=A2CNetwork)

for episode in count(1): # 一直加1的while, 表示一盘游戏，先得到20分胜

    # print('episode%d'%episode)

    obs, ep_return = agent_a2c.env.reset(), 0
    prev_x = None
    agent_a2c.rewards.append([])                # record rewards separately for each episode

    agent_a2c.log_probs_values.append([])

    for step in range(10000):
        curr_x = preprocess(obs)
        x = curr_x - prev_x if prev_x is not None else np.zeros(D)
        # 输入的其实是差分

        prev_x = curr_x
        action = agent_a2c.select_action(x)
        action_env = action + 2    # 不考虑static动作，模型输出的是0，1， 环境中变成2，3，分别对应上、下

        obs, reward, done, _ = agent_a2c.env.step(action_env)

        agent_a2c.rewards[-1].append(reward)
        ep_return += 1 if reward == 1 else 0

        if done:
            running_returns.append(ep_return)
            break


    if episode % batch_size == 0:       # 每20盘计算更新一次梯度

        agent_a2c.train()


    if episode % 20 == 0:
        # clear_output(True)

        plt.plot(pd.Series(running_returns).rolling(100, 20).mean())
        plt.title('episide:{}, time:{},'.format(episode, timedelta(seconds=int(timer()-start))))
        plt.show()
    if np.array(running_returns)[-10:].mean() > 10:    # 降低一点标准看看能不能跑通

        eval_return = agent_reinforce.eval_()
        print("Solved! eval return is now {}!".format(eval_return))
        break 
# =============================================================================
```

最后放一张对各种policy gradient的总结图:

![44](https://pic.downk.cc/item/5f7ebe1b1cd1bbb86baec4a1.png)

## 对policy gradient的理解

这一段本来应该放在前面一点来写的（放在时序因果后面），但是最终还是放在最后来写了，算是一个小总结。我们还是来看policy gradient的公式：

$$
\nabla_{\theta}J(\theta)=E_{\tau\sim\pi_{\theta}(\tau)}[\sum_{t=0}^{T-1}G_t\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)]
$$

它和我们常见的score function相比多了一个$G_t$，这其实就是强化学习和有监督学习的一个差异：有监督学习有label，强化学习没有。举个例子说明一下：

假设现在算出来$\pi_{\theta}$是这样的：

$$
\pi_{\theta}(s_t)=
\begin{cases}
0.2, & if~~a_t=1 \\
0.3, & if~~a_t=2 \\
0.5, & if~~a_t=3 
\end{cases}
$$

真实执行的动作是$a_t=3$，那么在监督学习里，$s_t$就会有一个label：$\begin{bmatrix} 0 \\\\0 \\\\ 1\end{bmatrix}$，然后我们用cross-entropy loss来表示它们之间的差异$CE_{loss}(\pi_{\theta}(s_t),label)=\log\pi_{\theta}(a_t\|s_t)=\log0.2 \times 0+\log0.3 \times 0+\log0.5\times1$，但是在强化学习中，这个label并不是一定是最优策略产生的动作（应该可以说一定不是），所以不一定是正确的label。于是我们使用$G_t$对梯度做一个加权，$G_t$大的就是好的action，我们鼓励策略去产生能够得到更大reward的轨迹，就像下图中展示的那样。

![45](https://pic.downk.cc/item/5f7ebe5a1cd1bbb86baed38e.png)

最后再说一点关于policy gradient里的loss：$G_t\log\pi_{\theta}(a_t\|s_t)$。

这个loss和我们经常提到的mean square loss、cross-entropy loss的功能不太一样，**it doesn't measure performance**。它是为了计算$J_{\theta}(\tau)$的梯度而构造出来的一个`loss`，但它本身的值并不能反映模型训练的好坏，你可能会看到你的loss到了$-\infty$，而你的策略表现得很差，所以在policy gradient的训练中，我们应该只关心average return，loss means nothing。

好了，就写到这了，下次介绍policy gradient的进阶。