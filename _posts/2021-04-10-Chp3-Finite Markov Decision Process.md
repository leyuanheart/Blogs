---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter3
subtitle:   Finite Markov Decision Process
date:       2021-04-10
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

![10](https://pic.imgdb.cn/item/60fbb0765132923bf85acc67.png)

$S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\ldots$

我们用$R_{t+1}$而 不是$R_t$来表示$A_t$导致的收益，是为了强调下一时刻的收益和下一时刻的状态是被环境一起决定的。

智能体和环境的界限划分仅仅决定了智能体进行**绝对控制**的边界，而并不是其知识的边界。（The agent–environment boundary represents the limit of the agent’s absolute control, not of its knowledge.）

使用收益信号来形式化目标是强化学习最显著的特征之一。

That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

至关重要的一点是：<font color=red >**我们设立收益的方式要能真正表明我们的目标**</font>。

In particular, the reward signal is <font color=red >not</font> the place to impart to the agent prior knowledge about <font color=red >how</font> to achieve what we want it to do.

例如，在国际象棋中，智能体只有当最终获胜时才能获得收益，而并非达到某个子目标，比如吃掉对方的子。如果实现这些子目标也能得到收益，那么智能体可能会找到某种即使绕开最终目的也能实现这些子目标的方式。例如，它可能会找到一种以输掉比赛为代价的方式来吃掉对方的子。

The reward signal is your way of communicating to the agent <font color=red >what</font> you want achieved, not <font color=red >how</font> you want it achieved.

要传授这种先验知识，更好的方法是设置初始的策略，或者初始的价值函数，或者对这二者施加影响。

## 分幕式和持续性任务

对于**分幕式**（Episodic）任务，每一幕是有终止时刻**$T$**的，对应的是一个特殊的状态**终结状态**。这里有一个小概念要提一下：

即使结束的方式不同，例如比赛的胜负，下一幕的开始状态的分布于上一幕的结束方式完全无关。因此，<font color=red >这些幕可以被认为是在同样的终结状态下结束的</font>，只是对不同的结果有不同的收益。

也就是说，我们不把赢或者输看成两种不同的状态，而是统一到$S_T$这样一个抽象的状态，只是经由$A_{t-1}$后得到的收益$R_T$是不一样的。

在分幕式任务中，我们有时需要区分非终结状态集$S$和包含终结状态的所有状态集$S^+$。终结时间$T$是一个随机变量，通常随着幕的不同而不同。

我们通常把强化学习的目标定义为最大化期望折后回报：

![11](https://pic.imgdb.cn/item/60fbb18b5132923bf85f6155.png)

如果我们定义$G_T=0$，那么上式会适用于任何时刻$t<T$，即使最终时刻出现在$t+1$也不例外。

我们可以将分幕式和持续性任务统一起来表示。把幕的终止当作一个特殊的**吸收状态**的入口，它之后转移到自己并且值产生0收益。

![12](https://pic.imgdb.cn/item/60fbb18b5132923bf85f6167.png)

这样一来，无论我们是计算前$T$个收益的总和，还是计算无限序列的总和，结果都是一样的。即使引入折扣，也依然成立。

![13](https://pic.imgdb.cn/item/60fbb18b5132923bf85f617c.png)

上式包括$T=\infty$或者$\gamma=1$（但不是二者同时）的可能性。

## 策略和价值函数

**<font color=red >终止状态的价值始终为0</font>**。

## Gridworld

![14](https://pic.imgdb.cn/item/60fbb18b5132923bf85f6198.png)

这个网格问题的例子。假设智能体在所有状态下选择而4种动作的概率都是相等的。上图右边就展示了这个策略下折扣系数$\gamma=0.9$时的价值函数$v_{\pi}$。这里面有几个点展示了即期收益和长期回报之间的关系：

- 状态$A$在这个策略下是最好的状态，但是其期望回报价值小于10，也就是说比当前时刻的即时收益要小。这是因为从状态$A$出发只能移动到$A'$，继而有很大可能出界（$A'$的价值是负数），获得$-1$的收益。
- 状态$B$的价值比当前时刻的即时收益（5）要大，这是因为从$B$到$B'$之后，$B'$的价值是一个正数（从$B'$出发时出界的期望惩罚（负收益值）小于从$B'$出发走到状态$A$或者$B$的期望收益值）。

在这个例子中，到达目标时收益为正数，出界时收益为负数，其他情况均为0。但是其实这些收益的符号并不是最重要的（当然，在某些情况下是很重要的），真正重要的是**<font color=red >收益的相对大小</font>**。

对于持续性任务，如果每个收益值都加上一个常数$c$，那么

![15](https://pic.imgdb.cn/item/60fbb18b5132923bf85f61ab.png)

在任何策略下，任何状态之间的相对价值不受其影响。

但是如果是分幕式任务，因为会有终止时刻$T$

![16](https://pic.imgdb.cn/item/60fbb1b45132923bf8601037.png)

状态之间的相对价值会受终止时刻$T$和折现因子$\gamma$的影响，举个例子：

![17](https://pic.imgdb.cn/item/60fbb1b45132923bf860105b.png)

而且如果想要智能体尽快完成任务，那么对于一般操作就要设置一个负的收益。

## 最优策略和最优价值函数

在本质上，价值函数定义了策略上的一个偏序关系。（Value functions define a partial ordering over policies.）

如果最优价值函数$v^\*$​​​​已知，那么单步搜索后最好的动作就是全局最优的动作。也就是说，对于$v^\*$​​​​来说，任何**贪心**策略都是最优策略。这是因为$v^\*$​​​​本身已经包含了未来所有可能的行为所产生的回报影响。定义$v^\*$​​​​的意义就在于，**<font color=red >我们可以将最优的长期（全局）回报值转化为每个状态对应的一个当前局部量的计算</font>**。

在给定$q^\*$​​的情况下，选择最优动作变得更加容易。甚至不需要进行单步搜索，只要找到使得$q^\*(s,a)$​​最大的$a$​​就行了，因为动作价值函数保存着所有单步搜素的结果。最优动作的选取不再需要知道后继状态和其对应的价值了，也就是说，**<font color=red >不再需要知道任何环境的动态变化特性了</font>**。

强化学习**<font color=red >在线运行</font>**的本质使得它在近似最优策略的过程中，对于那些经常出现的状态集合会花更多的努力去学习好的策略，而代价就是对不经常出现的状态给予的学习力度不够。这是区分强化学习和其他解决MDP问题的近似方法的一个重要的判断依据。

The online nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states. This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs.