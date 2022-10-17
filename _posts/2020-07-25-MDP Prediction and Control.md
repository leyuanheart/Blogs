---
layout:     post
title:      Reinforcement Learning 2
subtitle:   MDP Prediction and Control
date:       2020-07-25
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

先给出上回说到的两个重要公式：


$$
\begin{gather}
v^{\pi}(s)=\sum_{a \in A}\pi(a|s) \big( R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v^{\pi}(s') \big) \\
q^{\pi}(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a) \sum_{a' \in A}\pi(a'|s') q^{\pi}(s', a')
\end{gather}
$$


在MDP中，我们主要关注两类问题：

1. Prediction（评估一个给定的策略）
 - 输入一个MDP（$S,A,R,P,\gamma$）和策略$\pi$
 - 输出：value function $v^{\pi}$
2. Control（寻找最优的策略）
 - 输入一个MDP（$S,A,R,P,\gamma$）
 - 输出：最优value function $v^\*$，最优策略$\pi^\*$

当转移概率和reward function已知的情况下，可以用**动态规划**来求解上述两个问题。

关于动态规划，我了解的不多，我现在的理解就是一个迭代过程，但是它必须满足一些条件才能使用，这里就贴出周老师课程中的解释：
![dynamic-programming](https://pic.downk.cc/item/5f1b8cf014195aa59452b29e.png)

> 补充：解释一下上面两个特征，参考这个网站 [What is Dynamic Programming](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews/m2G1pAq0OO0)
>
> 以Fibonacci数列为例
>
> ![fibonacci](https://pic.downk.cc/item/5f1b8d7514195aa59452eb07.png)
>
> 我们知道$F(n)=F(n-1)+F(n-2)$ for $n>2$，这就是第一个特征 `optimal substructure`，$F(n)$的问题可以被拆成$F(n-1)$和$F(n-2)$，然后看图可以发现，为了解$F(4）$，$F(2)$被重复调用了2次，$F(1)$被重复调用了3次，这就是第二个特征`overlapping subproblems`，子问题的结果会重复多次利用。

## Prediction
目标：在一个MDP中评估一个策略$\pi$

方法：迭代Bellman Expectation Equation

算法： 

1. 在每一个t+1步
 - 更新$v_{t+1}(s)$ for all $s \in S$，利用R、P和$v_t(s')$，$s'$是$s$下一个state

   
   $$
   v_{t+1}(s)=\sum_{a \in A}\pi(a|s) \big(R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_t(s') \big)
   $$
   
2. 
   迭代直到收敛：$v_1 \to v_2 \to \ldots \to v^{\pi}$

## Control
首先定义一下最优value function和最优策略。


$$
v^*(s) = \underset{\pi}{\max}v^{\pi}(s)
$$



$$
\pi^*(s) = \underset{\pi}{\arg \max}v^{\pi}(s)
$$



如果我们知道一个MDP的optimal value，我们就说这个MDP is “solved”。

对于MDP，存在唯一的optimal value function，但是可能存在多个optimal policies。

那如何找到一个最优的策略呢，一种找法就是每一步$s$都选择使$q^*(s,a)$ 最大的那个$a$，这就是一个deterministic policy。


$$
\pi^*(a|s)=
\begin{cases}
1, & \text{if }a=\arg \underset{a \in A}{\max} q^*(s, a) \\
0, & \text{otherwise}
\end{cases}
$$


再次提醒$v^{\*}(s)$， $q^{\*}(s, a)$ 两者知道其中一个就可以推出另一个。

`对于MDP来说，总是存在一个确定性的最优策略。`

最原始的方法就是穷举所有的策略然后选出最优的（我才知道“穷举”的英文是“enumerate search”），但是穷举的数量是$\|A\|^{\|S\|}$，在状态空间和动作空间很大时显然不可取，接下来就介绍两种常用的方法：policy iteration和value iteration。

### Policy Iteration
分两步：

1. 评估当前的policy（计算当前policy的value function）
2. 改善当前的policy
$\pi'=greedy(v^{\pi})$，这个greedy就是上面说的在每一个state都采取使$q(s,a)$最大的action

放上周老师的图：
![5](https://pic.downk.cc/item/5f12a9f014195aa59467e25a.png)

具体的算法步骤：

1. 利用Bellman expectation equation迭代计算value function
   
   $$
   v_{t+1}(s)=\sum_{a \in A}\pi(a|s) \big(R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_t(s') \big)
   $$
   

2. 计算policy $\pi_i$的state-action value：
   
   $$
   q^{\pi_i}(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s, a)v^{\pi_i}(s')
   $$

   这里也可以写成$q^{\pi_i}(s,a)$和$q^{\pi_i}(s',a')$的关系式，可以自己试着写一下，答案在文章开头。

3. 计算$\pi_{i+1}$:
   
   $$
   \pi_{i+1}=\underset{a}{\arg \max}~q^{\pi_i}(s,a)
   $$

![6](https://pic.downk.cc/item/5f1b8d9d14195aa59452fe59.png)

再放一个周老师的证明，证明的是用greedy得到的策略一定是优于之前的策略的（value function会更高），公式太长，为了节约时间就直接放图了，比较重要的点我用红线标注出来的了。
![7](https://pic.downk.cc/item/5f1b8d9d14195aa59452fe65.png)

如果一直improve，直到收敛，


$$
q^{\pi}(s,\pi'(s))=\underset{a \in A}{\max}q^{\pi}(s,a)=q^{\pi}(s,\pi(s))=v^{\pi}(s)
$$


于是又诞生出又一个Bellman equation的版本，**Bellman optimality equation**


$$
v^{\pi}(s)=\underset{a \in A}{\max}q^{\pi}(s,a)
$$


这同样是一个非常非常重要的公式，后面的value iteration就要用到这个式子。

按照**Bellman Expectation Equations**的格式，再重新整理一下**Bellman Optimality Equations**:


$$
\begin{gather}
v^*(s)=\underset{a}{\max}q^*(s,a) \\
q^*(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v^*(s')
\end{gather}
$$


互相带入，就得到


$$
\begin{gather}
v^*(s)=\underset{a}{\max} \big(R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)v^*(s') \big)\\
q^*(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)\underset{a'}{\max}q^*(s',a')
\end{gather}
$$


如果大家还记得，这里求max的地方在**Bellman expectation equation**里都是求均值，这是它们两者的差别。

### Value Iteration

value iteration的思想就是，如果我们能

得到每一个subproblem的解$v^\*(s')$，那么最优的$v^\*(s)$就可以利用Bellman optimality equation来迭代获得：


$$
v(s) \leftarrow \underset{a \in A}{\max} \big( R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v(s')\big)
$$


和policy iteration不同的是，它并没有evaluate一个给定的policy（或者说它在迭代的过程中不通过value函数产生policy），而是基于当前的value function直接去优化，最后再从最优的value函数中产生policy。

算法流程：

1. initialize k=1，$v_0(s)=0$ for all states s
2. for k=1:H
   1. for each state s
      
      $$
      \begin{gather}
      q_{k+1}(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_k(s')\\
      v_{k+1}(s)=\underset{a}{\max}q_{k+1}(s,a)
      \end{gather}
      $$
   2. $k \leftarrow k+1$
3. to retrive the optimal policy after the value iteration:
   

   $$\pi(s)=\underset{a}{\arg \max} \big(R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_{k+1}(s') \big)$$

### PI和VI的比较
1. policy iteration包含两个过程：policy evaluation和policy improvement，两者交替进行直到收敛，然后从最优value函数中做policy extraction
2. value iteration是finding optimal value function + one policy extraction，其实finding optimal value function也可以看成是policy improvement（求max的步骤）和truncated policy evaluation（它也计算了q(s, a)，但是它并不是一直计算到收敛，而是只计算一步就重新指派v(s)了）。

下面是一张总结了使用动态规划来求解MDP中Prediction和Control问题的图，不同的问题使用了什么equation，用了什么算法：
![8](https://pic.downk.cc/item/5f1b8d9d14195aa59452fe6d.png)

### PV和VI的python实现
接下来就要上代码来实际操作了，还是沿用动作空间和状态空间都是离散的设定，所以选择了另一个游戏`FrozenLake-v0`，先说明一下这个游戏的规则：
![12](https://pic.downk.cc/item/5f1b8df914195aa594532d73.png)

如图所示，`FrozenLake-v0`是一个$4\times4$格子组成的冰面，格子上的字母分别表示：

- S: initial stat 起点
- F: frozen lake 冰湖
- H: hole 窟窿
- G: the goal 目的地

agent 要学会从起点走到目的地，并且不要掉进窟窿，如果走到目的地或者掉进冰窟窿里，游戏结束，达到目的地获得reward 1，其他情况均为0。

然后还有一点需要注意的就是，它的在某一个格子上的action（上[3]、下[1]、左[0]、右[2]）产生的结果并不是确定的，**因为冰面是滑的**，所以比如说当你采取向上走的动作，你并不一定到达上面的那个格子，而是会等可能的滑到3个格子中的一个，当然你也可以通过`env = gym.make('FrozenLake-v0', is_slippery=False)`来设置不滑动。

```python
import gym
env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'

env = gym.make(env_name)

env.action_space, env.observation_space
# out: (Discrete(4), Discrete(16))

env.env.nS,  env.env.nA
# out: (16, 4)
```

这个游戏还有一个`env.env.P`，这个就相当于环境模型，它会输出一个字典，里面包含16个状态，在每个状态里，有4个action，在每个action里，会有3个它可能会到达的state，获得的reward，以及done的值。

```python
env.env.P[0]. # 初始位置，一共有16个

{0: [(0.3333333333333333, 0, 0.0, False),
  (0.3333333333333333, 0, 0.0, False),
  (0.3333333333333333, 4, 0.0, False)],
 1: [(0.3333333333333333, 0, 0.0, False),
  (0.3333333333333333, 4, 0.0, False),
  (0.3333333333333333, 1, 0.0, False)],
 2: [(0.3333333333333333, 4, 0.0, False),
  (0.3333333333333333, 1, 0.0, False),
  (0.3333333333333333, 0, 0.0, False)],
 3: [(0.3333333333333333, 1, 0.0, False),
  (0.3333333333333333, 0, 0.0, False),
  (0.3333333333333333, 0, 0.0, False)]}
```

有了这个`env.env.P`，相当于有了一个后门，就不用一步步和环境交互去获得下一步的state，reward什么的了，并不是每个游戏都有这样一个后门的，应该是只有离散状态和离散动作才会有这样一个table形式的环境模型。

#### Policy Iteration
算法再放一遍：

具体的算法步骤：

1. 利用Bellman expectation equation迭代计算value function
   
   $$
   v_{t+1}(s)=\sum_{a \in A}\pi(a|s) \big(R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_t(s') \big)
   $$
   

2. 计算policy $\pi_i$的state-action value：
   
   $$
   q^{\pi_i}(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s, a)v^{\pi_i}(s')
   $$

   

3. 计算$\pi_{i+1}$:
   
   $$
   \pi_{i+1}=\underset{a}{\arg \max}~q^{\pi_i}(s,a)
   $$


```python
import gym
import numpy as np
env = gym.make('FrozenLake-v0')

# =========policy evaluation: compute the value function given the policy =====

def evaluate_policy(env, policy, gamma=1):
    eps = 1e-10
    v = np.zeros(env.env.nS)
    # 由于环境已知，所以可以直接进行交互得到对应的reward和下一个状态 
    
    while True:
        pre_v = np.copy(v)
        for s in range(env.env.nS):
            a = policy[s]
            v[s] = sum([p * (r + gamma * v[_s]) for p, _s, r, _ in env.env.P[s][a]])  
        '''
        需要稍微注意一下，这里的sum并不是上面公式里的对$\pi(s|a)$求平均，
        因为这里的policy是deterministic的，一个状态下只会采取一个action，
        这个sum是里面那个转移概率求平均
        '''
        if (np.sum(np.fabs(v-pre_v)) <= eps):    # np.fabs: elementwise 计算绝对值
        
            break
    return v
    
# =============policy improvement ==================

def extract_policy(env, v, gamma=1):
    q_sa = np.zeros((env.env.nS, env.env.nA))
    for s in range(env.env.nS):
        for a in range(env.env.nA):
            q_sa[s][a] = sum([p * (r + gamma * v[_s]) for p, _s, r, _ in env.env.P[s][a]])
    new_policy = np.argmax(q_sa, axis=1)
    return new_policy

# 改成向量来存储q_sa会更高效

def extract_policy(env, v, gamma=1):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (prob, next state, reward, done)
                p, _s, r, _ = next_sr   
                q_sa[a]  += p * (r + gamma * v[_s])
        policy[s] = np.argmax(q_sa)
    return policy
    
# ======= policy iteration =============================

def policy_iteration(env, gamma=1):
    max_iter = 20000
    new_policy = np.random.choice(env.env.nA, size=(env.env.nS)) # 随机初始化一个策略
    
    for i in range(max_iter):
        old_policy = new_policy
        v = evaluate_policy(env, old_policy, gamma)
        new_policy = extract_policy(env, v, gamma)
        if (np.all(new_policy == old_policy)):
            print('policy iteration converge at %d' % (i+1))
            break
    print(new_policy)
    return new_policy 
    
# ============= 运用policy来跑游戏，观察收益 ================

def roll_out(env, policy, gamma=1, render=False):
    obs = env.reset()
    total_revenue = 0
    step = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))  # 这里其实要看你的policy输出的是整形还是浮点型来相应的调节
        
        total_revenue += gamma**step * reward     # 很多时候就直接加了，不折现
        
        step += 1
        if done:
            break
    return total_revenue
    
def batch_evaluate_policy(env, policy, gamma=1, n=100):
    scores = [roll_out(env, policy, gamma, render=False) for _ in range(n)]
    return np.mean(scores)
    
# ============= training and evaluation ====================

optimal_policy = policy_iteration(env, gamma=1)
scores = batch_evaluate_policy(env, optimal_policy, gamma = 1.0)
print('Average scores = ', scores)

# out:

# policy iteration converge at 4

# [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]

# Average scores =  0.69
```

#### Value Iteration
算法流程：

1. initialize k=1，$v_0(s)=0$ for all states s
2. for k=1:H
   1. for each state s
      
      $$
      \begin{gather}
      q_{k+1}(s,a)=R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_k(s')\\
      v_{k+1}(s)=\underset{a}{\max}q_{k+1}(s,a)
      \end{gather}
      $$
   2. $k \leftarrow k+1$
3. to retrive the optimal policy after the value iteration:
   

   $$\pi(s)=\underset{a}{\arg \max} \big(R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)v_{k+1}(s') \big)$$

```python
import gym
import numpy as np
env = gym.make('FrozenLake-v0')

# ============== # 运用Bellman optimality equation来迭代 =====================

def value_iteration(env, gamma=1):
    eps = 1e-10
    v = np.zeros(env.env.nS)
    q = np.zeros(env.env.nA)
    max_iter = 20000
    for i in range(max_iter):
        v_old = np.copy(v)
        for s in range(env.env.nS):
            for a in range(env.env.nA):
                q[a] = sum([p * (r + gamma * v_old[_s]) for p, _s, r, _ in env.env.P[s][a]])
            v[s] = np.max(q)
            
        if (np.sum(np.fabs(v-v_old)) <= eps):
            print('value iteration converged at iter: %d' % (i+1))
            break
    # print(v)
    
    return v
    
# 下面这个写法更高效，可以不用创建q，而且注意这里和前面的policy iteration中的policy evaluation不同，这里是要计算s下所有的a，然后max，而之前是通过policy产生a再更新

def value_iteration(env, gamma=1):
    eps = 1e-10
    v = np.zeros(env.env.nS)
    max_iter = 20000
    for i in range(max_iter):
        v_old = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + gamma * v_old[_s]) for p, _s, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)    
        if (np.sum(np.fabs(v-v_old)) <= eps):
            print('value iteration converged at iter: %d' % (i+1))
            break          
    # print(v)
    
    return v

# =================接下来的代码是之前policy iteration里用过的， extract_policy和在线玩游戏=======

def extract_policy(env, v, gamma=1):
    q_sa = np.zeros((env.env.nS, env.env.nA))
    for s in range(env.env.nS):
        for a in range(env.env.nA):
            q_sa[s][a] = sum([p * (r + gamma * v[_s]) for p, _s, r, _ in env.env.P[s][a]])
    new_policy = np.argmax(q_sa, axis=1)
    return new_policy

# 改成向量来存储q_sa会更高效

def extract_policy(env, v, gamma=1):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (prob, next state, reward, done)
                
                p, _s, r, _ = next_sr   
                q_sa[a]  += p * (r + gamma * v[_s])
        policy[s] = np.argmax(q_sa)
    return policy

# ============= 运用policy来跑游戏，观察收益 ================

def roll_out(env, policy, gamma=1, render=False):
    obs = env.reset()
    total_revenue = 0
    step = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))  # 这里其实要看你的policy输出的是整形还是浮点型来相应的调节
        
        total_revenue += gamma**step * reward     # 很多时候就直接加了，不折现
        
        step += 1
        if done:
            break
    return total_revenue
    
def batch_evaluate_policy(env, policy, gamma=1, n=100):
    scores = [roll_out(env, policy, gamma, render=False) for _ in range(n)]
    return np.mean(scores)
    
# ============= training and evaluation ====================

optimal_value = value_iteration(env, gamma=1)
optimal_policy = extract_policy(env, optimal_value, gamma=1)
scores = batch_evaluate_policy(env, optimal_policy, gamma = 1.0)
print(optimal_policy)
print('Average scores = ', scores)

# out:

# value iteration converged at iter: 877

# [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]

# Average scores =  0.64
```

可以看到两种方法得到的策略最后都是一样的，在线评估的结果由于环境随机性的原因不一样很正常。

> 还有我发现在本地写markdown显示的公式和上传到服务器山显示出来的不一样，尤其是行内的公式，原因是行内公式如果涉及 `(*,|)` 这些markdown里的具有作用的符号，写在公式里就必须要用 `\` 转义一下。

好了，这篇文章就写到这，下一次介绍model free的Prediction和Control。
