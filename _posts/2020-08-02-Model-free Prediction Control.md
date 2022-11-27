---
layout:     post
title:      Reinforcement Learning 3
subtitle:   Model-free Prediction and Control
date:       2020-08-02
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

上一篇我们介绍了动态规划求解一个MDP的prediction和control问题，回顾一下

![13](https://pic.downk.cc/item/5f2610f314195aa5945cf8f7.png)

![14](https://pic.downk.cc/item/5f2610f314195aa5945cf8f9.png)

但是之所以能这么做是因为我们对这个MDP的转移概率和奖励函数是完全知道的，这样其实相当于有一个后门，强化学习不需要和环境交互来进行学习了。但是在现实中很多时候环境是非常复杂的，就算将其建模成MDP，我们很可能是不知道转移概率和奖励函数的，或者是这个机制太复杂以至于难以利用，比如Atari游戏，围棋、机器人控制等等。

所以model-free的方法就次登场，在不知道环境模型的基础上通过和环境进行交互来学习（这其实也是我们通常意义下所指的强化学习）。主要有两种方法：Monte-Carlo policy evaluation和Temporal-Difference Learning。

## Monte-Carlo policy evaluation

我们的目标是估计$v^{\pi}(s)=E_{\pi}[G_t\|S_t=s]$，其中$G_t=R_{t+1}+\gamma R_{t+2}+\ldots$ under policy $\pi$。

MC policy evaluation的思想就是：采样很多条轨迹，计算实际的returns，然后把它们平均。样本均值估计总体均值，简单朴素的思想。

所以MC其实不仅不需要知道MDP的转移概率和奖励，甚至都不需要MDP这个假设，不过它也有使用限制，就是只能用于episodic 环境（每条轨迹必须有终止）。

### 算法

为了估计$v^{\pi}(s)$,

1. 对于一条轨迹中的状态$s$出现的每一个time step $t$（这里也可以只取第一个，分别对应every-visit和first-visit）

2. 计算incremental counter $N(s)\leftarrow N(s)+1$
3. 计算incremental return $R(s)=R(s)+G_t$
4. 估计$v(s)=R(s)/N(s)$

由大数定律，当$N(s)\to\infty$，$v(s)\to v^{\pi}(s)$

这里还有一个小技巧，叫做`incremental mean`


$$
\mu_t = \frac{1}{t}\sum_{j=1}^{t}x_j=\frac{1}{t}\left(x_t+\sum_{j=1}^{t-1}x_j \right)=\frac{1}{t} \left( x_t+(t-1)\mu_{t-1} \right)=\mu_{t-1}+\frac{1}{t}(x_t-\mu_{t-1})
$$


所以算法修改成：

1. 收集轨迹$(S_1,A_1,R_1,\ldots,S_T)$

2. 对于每个state $S_t$计算$G_t$

$$
N(S_t)\leftarrow N(S_t)+1\\
v(S_t)\leftarrow v(S_t)+\frac{1}{N(S_t)}(G_t-v(S_t))
$$

对于一些不平稳（non-stationary）的问题（就是环境可能会随着时间变化），其实大部分问题都是这样的，可以考虑使用running mean（old values are forgotten）：$v(S_t)\leftarrow v(S_t)+\alpha(G_t-v(S_t))$

### MC policy evaluation 代码

还是以FrozenLake-v0举例

![12](https://pic.imgdb.cn/item/6382c93a16f2c2beb12571be.jpg)

```python
import gym
import numpy as np
# import collections

env = gym.make('FrozenLake-v0')


# ================== running mean ===========================

values = np.zeros(env.nS)
alpha = 0.01
policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]    # 上一篇文章中的最佳策略

gamma = 1

for _ in range(5000):
    return_ = []
    rewards = []
    R = 0
    state = env.reset()
    states=[state]
    for _ in range(1000):
        a = policy[state]
        state, reward, done, _ = env.step(a)
        rewards.append(reward)
        states.append(state)
           
        if done:
            # print(len(states), len(rewards))
            
            for r in rewards[::-1]:     # 计算Return，从后往前算，是一个巧妙的设计！
                
                R = r + gamma * R
                return_.insert(0, R)
            for s, g in zip(states[:-1], return_):   # 这里要注意一下，终止状态没有参与评估，下面会详细介绍
                
                values[s] = values[s] + alpha * (g - values[s])
            break
print(values)
# ================== incremental mean ===========================

values = {0:[0, 0], 1:[0, 0], 2:[0, 0], 3:[0, 0], 4:[0, 0], 5:[0, 0],
          6:[0, 0], 7:[0, 0], 8:[0, 0], 9:[0, 0], 10:[0, 0], 11:[0, 0],
          12:[0, 0], 13:[0, 0], 14:[0, 0], 15:[0, 0]}
policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
gamma = 1

for _ in range(5000):
    return_ = []
    rewards = []
    R = 0
    state = env.reset()
    states=[state]
    for _ in range(1000):
        a = policy[state]
        state, reward, done, _ = env.step(a)
        rewards.append(reward)
        states.append(state)
           
        if done:
            # print(len(states), len(rewards))
            
            for r in rewards[::-1]:
                R = r + gamma * R
                return_.insert(0, R)
            for s, g in zip(states[:-1], return_):
                values[s][0] += 1
                values[s][1] = values[s][1] + (g - values[s][1]) / values[s][0]
            break 
print(values)
# ========================================================

# 首先是用上一篇的value iteration算出来的value值

# [0.82352941, 0.82352941, 0.82352941, 0.82352941,

# 0.82352941,  0.        , 0.52941176, 0.        , 

# 0.82352941,  0.82352941, 0.76470588, 0.        , 

# 0.        ,  0.88235294, 0.94117647, 0.        ]


# 然后running mean和incremental mean结果差不多，我这里就放running mean的结果

# [0.72480605, 0.38190325, 0.38711739, 0.28761858,

# 0.79230198,  0.        , 0.3109656 , 0.        ,

# 0.72189522,  0.71356237, 0.6621382 , 0.        , 

# 0.        ,  0.73649237, 0.87937573, 0.        ]


# 可以发现其实计算的不是很准了，而且波动很大，再运行一次就完全不是这个结果了，这也是MC的一个问题，

# 就是虽然单次采样的轨迹算出来的return是value函数的无偏估计，但方差太大，如果平均的次数不够是估不准的，我这已经跑了5000条轨迹了，仍然不够准。


# 另外还有一个问题，是我在实际写代码的时候发现的，之前看的时候根本就不会注意到，就是在更新value值的时候，关于终止状态的value值


# 首先结论是：终止状态是没有value值的（或者说是0），之前有说过关于轨迹中到底是用R_t还是R_{t+1}不用太在意，理解就好，但是现在发现不是这么回事，还是要理理清楚的，我们说一条轨迹严格意义上应该表示成：

# <S1, A1, R2, S2, A2, R3, A3, ..., S[T-1], A[T-1], R[T], S[T]>


# 在状态S1采取动作A1，然后环境接受后给一个反馈R2并且转移到下一个状态S2，如果从环境的角度看就是R2，如果从agent的角度看可以说是R1，

# 这两种理解在前面你随便怎么解释都没问题，但是到了终止状态就必须要注意了，R[T]是对应S[T-1]和R[T-1]的，这个不管是哪种理解都是一致的，在S[T]是不采取行动的，所以也不会有reward。


# 因此在存储对应的状态St和Gt的时候，要特别注意，由于env.step(a)产生的是下一时刻的状态和reward，如果用S[t+1],R[t+1]表示的话，用R[t+1]及以后的reward算出来因该是对应S[t]，

# 这在编写代码的时候就会出现states多一个的情况（记住一定要先append初始化的那个状态），因为每次执行完env.step(a)之后states和rewards都要append，这样就会把S[T]给加进去，所以在匹配的时候要去掉最后一个状态。


# 也不知道解释清楚没有，或者可能大部分人都没有这个问题，那这段就看个乐吧。
```

### MC和DP之间的区别

![15](https://pic.downk.cc/item/5f26118a14195aa5945d28ed.png)

动态规划（DP）是通过bootstrap当前所有的value值求期望来更新给定状态下的value，而MC是根据一次采样的样本均值来更新value。

## Temporal-Difference Learning policy evaluation

TD可以说是MC和DP的结合版，既用了sampling也用了bootstrapping。

**Bellman equation**告诉我们$v(S_t)=E_{\pi}[R_{t+1}+\gamma v(S_{t+1})]$，如果利用期望和$v(S_{t+1})$，那么就是纯bootstrapping，如果是用样本均值$G_t$来估，就是纯sampling，然后TD用的是$v(S_t)=R_{t+1}+\gamma v(S_{t+1})$，相当于实际向前走了一步，采样一个$R_{t+1}$，然后$S_{t+1}$的value用bootstrapping来估计。

因此TD(0)的算法流程是：


$$
v(S_t)\leftarrow v(S_t)+\alpha \big(R_{t+1}+\gamma v(S_{t+1})-v(S_t)\big)
$$


$v(S_t)=R_{t+1}+\gamma v(S_{t+1})$叫做`TD target`

$R_{t+1}+\gamma v(S_{t+1})-v(S_t)$叫做`TD error`

你可以向前走一步，也可以向前多走几步，这样就有了n-step TD


$$
G_t^n=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1}R_{t+n}+\gamma^nv(S_{t+n})
$$


n-step TD: $v(S_t)\leftarrow v(S_t)+\alpha \big( G_t^n-v(S_t)\big)$

![16](https://pic.downk.cc/item/5f26119f14195aa5945d2ec3.png)

你会发现当向前走无穷多步时，TD就变成了MC。

### TD($\lambda$)

这一块是补充的内容，之前没有学习到，还是在帮老师做课件的时候才发现，现在补上，里面有一些思想是挺有启发性的（虽然我觉得我还没有彻底理解）。关于这一部分的参考：

[强化学习入门 第四讲 时间差分法（TD方法)](https://zhuanlan.zhihu.com/p/25913410)

[强化学习（五）用时序差分法（TD）求解](https://www.cnblogs.com/pinard/p/9529828.html)

[机器学习（二十九）——Temporal-Difference Learning](https://blog.csdn.net/antkillerfarm/article/details/80305492)

上一节讲到，我们可以做n-step TD，就是说对于value函数，我们其实是有n个估计值，然后我们选择用哪一个，那能不能把这n个估计值都用起来呢，这就是TD($\lambda$)的想法，引入一个新的参数$\lambda$，通过加权的方法将n个估计值融合。

定义一个$\lambda$-return：


$$
G_t^{\lambda}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$


每一个$G_t^{(n)}$的权重是$(1-\lambda)\lambda^{(n-1)}$，这个权重的直观概念可以从下图可以看得比较明白，随着n的增大，第n步return的权重呈几何级数的衰减，当在T时刻达到终止状态时，未分配的权重全部给予中状态的实际return。所有权重加起来等于1，离当前状态越远的return权重越小。当$\lambda=0$时，就变成了TD(0)，当$\lambda=1$时，就变成了MC。

![td1](https://pic.downk.cc/item/5f4130f7160a154a67cd0c06.png)

然后就用$G_t^{\lambda}$来更新value函数：$V(S_t)\leftarrow V(S_t)+\alpha(G_t^{\lambda})-V(S_t)$，关于TD($\lambda$)，比较有意思的是可以从两种视角来看待。

#### 前向视角

这是一个正常的理解逻辑，由于$G_t^{\lambda}$中有$G_t^{(n)}$，而$G_t^n=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1}R_{t+n}+\gamma^nv(S_{t+n})$，所以其实也是需要等到整个episode结束之后才能更新的。那么是不是可以不需要等到episode结束就更新当前的value函数呢，这就要利用TD($\lambda$)的后向视角了。

![td2](https://pic.downk.cc/item/5f413169160a154a67cd39a1.png)

#### 后向视角

![td3](https://pic.downk.cc/item/5f413169160a154a67cd39a3.png)

先形象地解释一下这幅图表示的意思，就是说在当前时刻t处于状态$s_t$，采取action到达$s_{t+1}$之后，计算TD error，然后相当有一个人面朝已经经历过的状态喊话，告诉它们需要利用当前的TD error来进行更新，此时过往的每个状态value函数更新的大小应该更跟距离当前状态的步数有关。所以$s_{t-1}$处的更新应该乘以一个衰减因子$\gamma \lambda$，$s_{t-2}$处的更新应该乘以$(\gamma \lambda)^2$，以此类推。

![td4](https://pic.downk.cc/item/5f4131a4160a154a67cd5150.png)

这个对过去状态更新采用不同权重的做法可以理解成过去状态对当前状态的影响程度，影响大的自然要更新的多一些，比如老鼠在连续接受了3次响铃和1次亮灯后遭到了电击，那么在分析遭到电击的原因时，到底是响铃因素比较重要还是亮灯因素比较重要呢？这里就有两种考量：

1. 频率启发（frequency heuristic）：归因于过去发生次数多的因素
2. 就近启发（recency heuristic）：归因于最近发生的因素

为了同时利用上述的两种启发，对于每一个状态引入一个数值：**效用（eligibilty, E）**，而它随着时间变化的函数叫做**效用迹（eligibility trace, ET）**：



$$
\begin{gather}
E_0(s)=0 \\
E_t(s)=\gamma \lambda E_{t-1}(s)+I(s=S_t)
\end{gather}
$$


![td5](https://pic.downk.cc/item/5f413169160a154a67cd39a5.png)

上图是一个可能的效用迹的图，横坐标下面的竖线代表当前时刻t进入和状态s，可以看出，当状态s出现的次数增加1时，它的ET是会增加1的，然后随着时间的推移，它的ET会逐渐衰减，我们就用ET来作为每个时刻各个状态更新的权重。因此从后向视角来看，可以提供一种增量式的更新方法：$V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)$，具体的算法如下：

1. 计算当前状态的TD error：$\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$

2. 更新效用迹（注意这里是更新之前所有见过到过的状态）：

$$
E_t(s)=
\begin{cases}
\gamma \lambda E_{t-1}, &  \text{if } s \neq s_t \\
\gamma \lambda E_{t-1}+1, &  \text{if } s = s_t
\end{cases}
$$

3. 更新之前见到过得所有状态的value函数：$V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)$

#### 前向和后向的等价性

这里所说的等价性，指的是在一个episode结束后，每个状态的value函数的更新总量都是$G_t^{\lambda}-V(S_t)$，证明就不手打了，tricks我用红笔标出来了。

![td6](https://pic.downk.cc/item/5f4130f7160a154a67cd0c08.png)

### TD learning 代码

```python
import gym
import numpy as np
# import collections
env = gym.make('FrozenLake-v0')

# ================== running mean ===========================

values = np.zeros(env.nS)
alpha = 0.01
policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]    # slippery FrozenLake  optimal policy

# policy = [1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0]

gamma = 1

for _ in range(5000):
    # print(list(values))
    
    state = env.reset()
    for _ in range(2000):
        # print('every episode=====================')
        
        # print(list(values))
        
        a = policy[state]
        state_next, reward, done, _ = env.step(a)

        # TD update
        
        # 要注意如果是终止状态的话就直接是reward，没有后面的折现了
        
        target = reward + (1-done) * gamma * values[state_next]      # 一开始这里写成了1*done，找了半天不知道问题出在哪。。。心累啊
        
        values[state] = values[state] + alpha * (target - values[state])
        # print('state:{}'.format(state))
        
        state = state_next
        # print('next state:{}, reward:{}'.format(state, reward))
        
        if done:
            # print('===============================')
            
            break

# ================== n-step TD ===========================

values = np.zeros(env.nS)
alpha = 0.01
# policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]    # slippery FrozenLake  optimal policy

policy = [1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0]
gamma = 1
n = 1

for i in range(5000):
    state = env.reset()
    rewards = 0
    steps = 0
    state_ori = state
    # print("episode:{}==============".format(i))
    
    for _ in range(1000):
        a = policy[state]
        state_next, reward, done, _ = env.step(a)
        rewards += reward * gamma**steps
        steps += 1
        state = state_next
        if steps == n or done:
            # print(steps, done)
            
            target = rewards + gamma**steps * values[state_next] * (1-done)
            values[state_ori] = values[state_ori] + alpha * (target - values[state_ori])
            state_ori = state
            steps = 0
            rewards = 0
        
        if done:
            break
  
  
  # out
  
  [0.74539926, 0.63995419, 0.59800216, 0.57007509, 
  0.74807117,  0.        , 0.39433746, 0.        , 
  0.75278847, 0.76084139,  0.68410322, 0.        , 
  0.        , 0.83888238,  0.92888142, 0.        ]
  
```

### TD和MC评估的比较

![17](https://pic.downk.cc/item/5f2611d514195aa5945d3fe5.png)

### 关于MC和TD的一些思考（主要是写代码的时候发现的一些问题）

其实TD和MC都是涉及采样的方法，采样的 思想就是只有实际采样到的状态我才更新，没有采到的状态是不更新的，以FrozenLake来说明，因为环境的随机性，使得确定性策略产生的状态也具有随机性，也就是说在给定状态下同一个动作可能会到达不同的状态，所以对于给定的策略，进行采样的话，是有机会将所有的状态都给采样到的，那只要采样次数足够多，理论上也是能够达到动态规划的水平的，但是如果环境没有随机性，比如对于`不滑`的冰面，在某一个格子，采取一个动作，它是以概率1到到达另一个格子的，这个时候如果是要评估一个确定性的策略，用采样的方法就有问题了，因为从起始状态出发，只会走策略给出的指定路径，其他的一些格子根本走不到，这样这些格子的value是得不到更新的，还是以FrozenLake来说明，我把它设置成`is_slippery=False`，然后用DP去做value iteration可以得到最优的value值，但是用MC和TD去迭代就会发现有些格子的value值本该是1，结果却是0，原因就是刚才提到的，采样采不到这些状态，自然无法更新value，所以在用MC和TD做策略评估的时候需要考虑采样的随机性如何。  

另外还有一个问题，对于不滑的冰面，上面提到用DP算出了最优value值，是

|  1   |  1   |  1   |  1   |
| :--: | :--: | :--: | :--: |
|  1   |  0   |  1   |  0   |
|  1   |  1   |  1   |  0   |
|  0   |  1   |  1   |  0   |

然后你会发现，无法根据这个value表提取出最优策略，计算每个状态的4个动作的Q值，如果用的是`max`操作，默认是选第一个，由此组成的策略并不是最优的。。。这其实是我没有想到的，我还没有想的很明白为什么会这样，感觉一个可能是因为reward比较很稀疏，只有在到达终点时才获得1，其他的都是0（虽然我不知道怎么解释，就是感觉是这个问题）；另一个原因应该是`max	`这个操作的问题（但这也不是本质问题）。

>state: 0, q:[1. 1. 1. 1.]
>state: 1, q:[1. 0. 1. 1.]
>state: 2, q:[1. 1. 1. 1.]
>state: 3, q:[1. 0. 1. 1.]
>state: 4, q:[1. 1. 0. 1.]
>state: 5, q:[0. 0. 0. 0.]
>state: 6, q:[0. 1. 0. 1.]
>state: 7, q:[0. 0. 0. 0.]
>state: 8, q:[1. 0. 1. 1.]
>state: 9, q:[1. 1. 1. 0.]
>state: 10, q:[1. 1. 0. 1.]
>state: 11, q:[0. 0. 0. 0.]
>state: 12, q:[0. 0. 0. 0.]
>state: 13, q:[0. 1. 1. 1.]
>state: 14, q:[1. 1. 1. 1.]
>state: 15, q:[0. 0. 0. 0.]

对于这个不滑的FrozenLake，虽然是存在一个最优的确定性策略的，但是通过value-based的方法似乎找不出来，感觉如果用随机性策略会更好一些，可能用policy-based的方法（之后会介绍）可以，啊，这段写得很乱。。。。。。

## Model-free Control

上面讨论的是如何评估一个给定的策略，现在是寻找最优策略，但其实所谓model-free 的control，就是在评估阶段使用了model-free的方法，在improving的阶段还是和上一篇提到的policy iteration和value iteration一样。

### Generalized policy iteration with MC

<img src="https://pic.downk.cc/item/5f26122114195aa5945d51fe.png" style="zoom:50%;" />

![19](https://pic.downk.cc/item/5f26122114195aa5945d5201.png)

根据上图的介绍，可以看到在policy evaluation阶段用MC去估计Q(s, a), policy improvement阶段用greedy Q。

还有一点需要注意的是，这个算法要收敛是有条件的，其中一个就是：Episode has `exploring starts`（绿线标注的地方）。就是说初始化的时候要可以任意位置，任意动作的初始化，这样就能保证每一个状态和动作都能够被更新，这就对应了我上面提到的关于采样的随机性问题（哦，我突然想到之前应该随机化初始状态的，但是好像Gym做不到这一点。。。）

### $\epsilon$-Greedy Exploration

这一小节讲一对强化学习里很重要的概念：`Exploration（探索）`和`Exploitation（利用）`。

由于强化学习是一个序贯决策的问题，所以就会面临在当前时刻到底是应该利用目前最好的结果，还是去探索未知的动作，这需要做一个tradeoff。如果是动态规划，因为每一个子问题都能够等到最优解，然后构成全局最优，所以一直利用是没有问题的，但是现在model-free的情况下，每一步评估得到的Q(s, a)不再是最优的了（估不准，存在偏差），所以一直用greedy策略就可能会陷入局部最优解，而采样不到可能更好的动作。因此，一种增加探索的方法就是：**$\epsilon$-Greedy**:

- 以$1-\epsilon$的概率选择greedy动作
- 以$\epsilon$的概率随机选择一个动作

$$
\pi(a|s)=
\begin{cases}
\epsilon/|A|+1-\epsilon,~~~~if~a^*=\arg \underset{a\in A}{\max}Q(s,a) \\
\epsilon/|A|,~~~~~~~~~~~~~~~~~otherwise
\end{cases}
$$

**$\epsilon$-Greedy**既保证了策略是在improved，同时也加入了探索的成分，当然还有其他平衡两者的方法，比如UCB（Upper Confidence Bandit)，之后用到了再介绍。所以改进后的算法如下：

![20](https://pic.downk.cc/item/5f26127b14195aa5945d6d87.png)

### Sarsa: TD Control

把MC换成TD就变成了一个经典的强化学习算法：Sarsa。为什么叫这个名字呢，因为它利用的是（$S_t,A_t,R_{t+1},S_{t+1},A_{t+1}$）这个5元组。

![21](https://pic.downk.cc/item/5f26127b14195aa5945d6d8b.png)


$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]
$$


$R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$叫做TD target。

![22](https://pic.downk.cc/item/5f26127b14195aa5945d6d91.png)

#### Sarsa($\lambda$)

这里是接上面的TD($\lambda$)，直接放一下算法流程吧：

算法输入：迭代轮数$T$，状态集$S$, 动作集$A$, 步长$\alpha$，衰减因子$\gamma$, 探索率$\epsilon$, 多步参数$\lambda$

1. 随机初始化所有的状态和动作对应的价值$Q$. 对于终止状态其$Q$值初始化为0.

2. for i from 1 to T，进行迭代:

- 初始化所有状态动作的效用迹$E$为0，初始化$S$为当前状态序列的第一个状态。设置$A$为$\epsilon$−贪婪法在当前状态$S$选择的动作。

- - 在状态$S$执行当前动作$A$, 得到新状态$S'$和奖励$R$

- - 用$\epsilon$−贪婪法在状态$S'$选择新的动作$A'$

- - 更新效用迹函数$E(S,A)$和TD误差$\delta$:

$$
\begin{gather}
E(S,A)=E(S,A)+1 \\
\delta=R_{t+1}+\gamma Q(S',A')-Q(S,A)
\end{gather}
$$

- - 对当前序列所有出现的状态$s$和对应动作$a$, 更新价值函数$Q(s,a)$和效用迹函数$E(s,a)$:

$$
\begin{gather}
Q(s,a)=Q(s,a)+\alpha \delta E(s,a)   \\
E(s,a)=\gamma \lambda E(s,a)
\end{gather}
$$

- - $S=S'$, $A=A'$

- - 如果$S′$是终止状态，当前轮迭代完毕				

输出：所有的状态和动作对应的价值$Q$	

### On-policy vs. Off-policy Learning

又需要介绍一对强化学习里很重要的概念了。

on-policy learning是指：用策略$\pi$采集的数据来学习$\pi$。

off-policy learning是指：用不同于策略$\pi$的策略采集的数据来学习$\pi$。

本来on-policy learning看起来顺理成章，为什么会出现一个off-policy learning呢？

原因是on-policy learning的**采样效率非常低**，由于采样（同环境交互）和优化的是同一个策略，所以就必须是等采样完才能优化，优化完才能继续采样，大量的时间花在与采样上，导致效率低下，而且也会使策略相对保守（这个要根据具体问题来看是不是一个缺点，因为有些问题就是需要一个相对安全稳定的策略，比如无人车驾驶），而off-policy learning则可以改善这个问题，因为要学习的策略与采样的策略不是同一个，这样就可以重复利用很多之前不同策略采样的数据，而且还可以让采样的策略去更多的探索，而要学习的的策略则进行最大化的利用。下面这张图很形象地解释了什么是off-policy learning。

![23](https://pic.downk.cc/item/5f26129b14195aa5945d7658.png)

Sarsa就是on-policy learning，而接下来要介绍的Q-Learning则是off-policy learning。

### Q-Learning: Off-policy TD control

Sarsa的学习是基于$(S_t, A_t,R_{t+1},S_{t+1},A_{t+1})$，$A_t$和$A_{t+1}$都是来自于$\epsilon$-greedy策略，Q-learning与Sarsa的区别就在于`$A_{t+1}$`不是来自于$\epsilon$-greedy策略，而是来自于`greedy`策略。

Q-learning 关键点：

1. target policy $\pi$ is `greedy` on $Q(s,a)$
   
   
   $$
   \pi(S_{t+1})=\arg \underset{a'}{\max}Q(S_{t+1},a')
   $$

2. behavior policy $\mu$可以是完全随机的，但是我们让它也在改进，通过$\epsilon$-greedy on $Q(s,a)$

3. Q-learning target：
   $$
   R_{t+1}+\gamma Q(S_{t+1},A') = R_{t+1}+\gamma Q(S_{t+1}, \arg \underset{a'}{\max}Q(S_{t+1},a'))=R_{t+1}+\gamma \underset{a'}{\max}Q(S_{t+1},a')
   $$
   

4. Q-learning update:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \underset{a'}{\max}Q(S_{t+1},a')-Q(S_t,A_t)]
$$



![24](https://pic.downk.cc/item/5f2612b314195aa5945d7bf2.png)

### Sarsa和Q-learning的代码

玩的是`MountainCar`

![25](https://pic.downk.cc/item/5f2612b314195aa5945d7bf4.png)

```python
import numpy as np
import gym

name = "MountainCar-v0"
env = gym.make(name)

# env.observation_space, env.action_space

# env.observation_space.sample(), env.action_space.sample()

# env.observation_space.high, env.observation_space.low

"""
状态是一个2维连续变量【位置，速度】，动作是一个离散变量，取值 0（向左推）, 1（不动）, 2（向右推）
只有到达右边山顶的黄旗处得0.5，其他状态都是0
"""

off_policy = True # if True use q-learning, if False use Sarsa

n_states = 40
def obs_to_state(env, obs):
    """
    Map an observation to state
    discrete the continuous observation space
    """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b


t_max = 10000
def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    steps = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma**steps * reward
        steps += 1
        
        if done:
            break
    return total_reward


iter_max = 5000
initial_lr = 1.0
min_lr = 0.003
gamma = 1.0
eps = 0.1
env.seed(0)
np.random.seed(0)

if off_policy == True:
    print ('----- using Q Learning -----')
else:
    print('------ using SARSA Learning ---')

q_table = np.zeros((n_states, n_states, 3))

for i in range(iter_max):
    obs = env.reset()
    total_reward = 0
    # eta: learning rate is decayed every 100 steps
    
    eta = max(min_lr, initial_lr * (0.85**(i//100)))
    
    for j in range(t_max):
        a, b = obs_to_state(env, obs)
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(q_table[a][b])
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        # update q table
        
        a_, b_ = obs_to_state(env, obs)
        if off_policy == True:
            # q-learning
            target = reward + gamma * np.max(q_table[a_][b_]) * (1 - done)
            q_table[a][b][action] = q_table[a][b][action] + eta * (target - q_table[a][b][action])
        else:
            # Sarsa
            
            if np.random.uniform(0, 1) < eps:
                action_ = np.random.choice(env.action_space.n)
            else:
                action_ = np.argmax(q_table[a_][b_])
            target = reward + gamma * q_table[a_][b_][action_] * (1 - done)
            q_table[a][b][action] = q_table[a][b][action] + eta * (target - q_table[a][b][action])
            
        if done:
            break
        
    if i % 200 == 0:
        print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
        
    
solution_policy = np.argmax(q_table, axis=2)
solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
print("Average score of solution = ", np.mean(solution_policy_scores))

# Animate it

for _ in range(2):
    run_episode(env, solution_policy, True)
env.close()  
```

### Sarsa和Q-Learning的比较

放一张图

![26](https://pic.downk.cc/item/5f2612d614195aa5945d850d.png)

### 为什么Off-policy能用？

这个问题在我第一遍学的时候完全没有意识到这是一个问题，只是觉得，哦，off-policy能利用历史数据，这很好，但是仔细想一下就会发现，类比一下，可能不太恰当，off-policy learning就像是用狗的图片去训练一个猫的分类器，这么一比就觉得好像不太靠谱，所以它为什么work，其实是一个问题。

这里先简单介绍一下**Importance Sampling**，之后介绍policy-based RL的时候还会详细介绍。

Importance Sampling的思想用一个公式就可以表达：


$$
\mathbb{E}_{x\sim p(x)}[f(x)]=\int p(x)f(x)dx=\int q(x)\frac{p(x)}{q(x)}f(x)dx=\mathbb{E}_{x\sim q(x)}[\frac{p(x)}{q(x)}f(x)]
$$


Importance Sampling让我们可以用来自$q(x)$的数据去估计$f(x)$在$p(x)$下的期望，但是必须要对$x$进行一个权重的调整来纠正由分布不同导致的偏差。这就是off-policy learning可以work的原因，然后就又有一个问题了，你会发现Q-learning并没有乘上一个ratio来做矫正，也就是说，Q-learning没有用Importance Sampling，这又是为什么呢？

一句话说明：**Q-learning的更新并没有用到behavior policy的信息**。有些人就会问了，怎么没用到，$S_t$和$A_t$不就是behavior policy产生的吗？这就是问题的关键，如果是学习$Q(S_t，A_t)$，那其实我并不关心$S_t$和$A_t$是由什么策略采集得到的，因为$Q^{\pi}(s,a)$的定义就是在给定$s$和$a$后，采用策略$\pi$得到的累计收益的期望，$s$和$a$是怎样得到的并不会影响对$Q^{\pi}(s,a)$的更新，而$S_t,A_t \to S_{t+1}$是由环境决定的，所以整个更新式子中没有用到behavior policy，完全可以把它视为只有一个target policy在起作用，这样自然就不用做Importance sampling了。

## 总结

放两张周老师课件中的图，我觉得总结得非常清晰到位，都不用我再解释了。

![27](https://pic.downk.cc/item/5f2612f014195aa5945d8de9.png)

![28](https://pic.downk.cc/item/5f2612f014195aa5945d8deb.png)

下一篇就要开始写function approximation了，会用到Deep learning的知识，我争取代码用TensorFlow2和Pytorch两种框架都写一遍，所以进度可能会比较慢。

