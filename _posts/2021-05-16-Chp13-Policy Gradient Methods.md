---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter13
subtitle:   Policy Gradient Methods
date:       2021-05-16
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

In this chapter we consider methods that instead learn a **parameterized policy** that can select actions without consulting a value function. A value function may still be used to learn the policy parameter, but is not required for action selection.

在这一章，我们讨论直接学习参数化策略的方法，动作的选择不再直接依赖于价值函数。价值函数仍然可以用于学习策略的参数，但对于动作选择而言就不是必须的了。

<font color=red> In practice, to ensure exploration we
generally require that the policy never becomes deterministic</font>

## 策略近似及其优势

如果动作空间是离散的并且不是特别的大，自然的参数化方法是对每一个“状态-动作”二元组估计一个参数化的**数值偏好**（$h(s, a, \theta) \in \mathbb{R}$）。

![59](https://pic.imgdb.cn/item/60fbb9c45132923bf88231e6.png)

这些动作偏好值可以被任意地参数化。可以用神经网络表示，也可以是特征的简单线性组合。

有人肯定会想了：那是不是也可以用动作价值来代替这个动作值偏好呢？

当然，我们也可以根据动作价值的softmax分布来选择动作，但是会有一个问题，就是这样得到的策略不能趋向于一个确定性的策略。这里需要解释一下，这里的“不能”不代表我们希望它是一个确定性的策略（因为在有些情况下我们就是希望策略带有一些随机性，比如，在非完全信息的纸牌游戏中，最优的策略一般是以特定的概率选择两种不同的玩法 ），而是说它无法引导出一个接近于确定的策略。

原因就在于：**动作价值的估计会收敛于对应的真实值**，而这些真实值之间的差异是有限的，因此各个动作会对应到一个特定的概率值而不是0和1。If the soft-max distribution included a temperature parameter, then the temperature
could be reduced over time to approach determinism, but in practice it would be difficult to choose the reduction schedule, or even the initial temperature, without more prior knowledge of the true action values than we would like to assume.

而动作偏好则不同，<font color=red> 因为它们不会趋向于任何特定的值</font>；相反，它们趋向于最优的随机策略。如果最优策略是确定性的，则最优动作的偏好值也可能趋向无限大于所有次优的动作（如果参数允许的话）。

策略参数化相对于动作价值参数化的一个最简单的优势可能是策略可以用更简单的函数近似。（Problems vary in the complexity of their policies and action-value functions. For some, the action-value function is simpler and thus easier to approximate. For others, the policy is simpler.）

策略参数化形式的选择有时是在基于强化学习的系统中引入理想中的策略形式的先验知识的一个好方法。这也是我们一般使用基于策略的学习方法的最重要的原因之一。

## 带有动作切换的小型走廊任务

![61](https://pic.imgdb.cn/item/60fbba325132923bf8840641.png)

上图中的曲线的精确表达式可以通过如下方式求解。

![60](https://pic.imgdb.cn/item/60fbb9c45132923bf8823202.png)

## 策略梯度定理

With continuous policy parameterization the action probabilities change smoothly as a function of the learned
parameter, whereas in "-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values, if that change results in a different action having the maximal value.

对于连续的策略参数化，选择动作的概率作为被优化的参数的函数会平滑地变化，但是$\epsilon$-贪心方法则不同，只要估计的动作价值函数的变化导致了最大动作价值对应的动作发生了变化，则选择某个动作的概率就可能会突然变化很大，即使估计的动作价值只发生了任意小的变化。**很大程度上由于这个原因，基于策略梯度的方法能够比基于动作价值函数的方法有更强的收敛保证。**

在这一节，我们考虑分幕式情况，将性能指标定义为幕初始状态的价值。我们可以在不丧失任何泛化性的情况下进行一些符号的简化，假设每幕都是从某个（非随机）状态$s_0$开始。

![62](https://pic.imgdb.cn/item/60fbba325132923bf8840652.png)

With function approximation it may seem challenging to change the policy parameter in a way that ensures improvement. 主要问题是性能既依赖于动作的选择，<font color=red>也依赖于动作选择时所处的状态的分布</font>，而它们都会受策略参数的影响。给定一个状态，策略参数对动作选择及收益的影响可以根据参数比较直观地计算出来。但是因为状态分布和环境有关，所以策略对状态分布的影响一般很难确切知道。而性能指标对模型参数的梯度却依赖于这种未知的影响，这是一个问题。

Fortunately, there is an excellent theoretical answer to this challenge in the form of the **policy gradient theorem**, which provides an analytic expression for the gradient of performance with respect to the policy parameter that does not involve the derivative of the state distribution.

![63](https://pic.imgdb.cn/item/60fbba325132923bf8840669.png)

下框给出了分幕式情况下的证明。

![64](https://pic.imgdb.cn/item/60fbba325132923bf8840679.png)

 

## REINFORCE：蒙特卡洛策略梯度

策略梯度定理给出了一个正比于梯度的精确表达式。现在只需要一种采样方法，是的采样样本的梯度近似或等于这个表达式。注意策略梯度定理公式的右边是将目标策略$\pi$下每个状态出现的频率作为加权系数的求和项，如果按策略$\pi$执行，则状态将按这个比例出现。因此

![65](https://pic.imgdb.cn/item/60fbba325132923bf884069a.png)

We could stop here and instantiate our stochastic gradient-ascent algorithm as

![66](https://pic.imgdb.cn/item/60fbba5f5132923bf884c55a.png)

这里的$\hat{q}$是由学习得到的$q_{\pi}$地近似。这个算法被称为**全部动作**算法，因为它的更新涉及了所有可能的动作。

It is promising and deserving of further study, but our current interest is the classical REINFORCE algorithm
(Willams, 1992) whose update at time $t$ involves just $A_t$, the one action actually taken at time $t$.

接下来我们继续进行REINFORCE算法的推导。和上式用引入$S_t$​的过程类似，我们将$A_t$​引入进来，把对随机变量所有可能取值的求和运算替换为对$\pi$​的期望，然后对期望进行采样。但是我们会发现，上式中虽然有求和项，但是每一项并没有将$\pi(a\|S_t, \theta)$​作为加权系数，而这是对$\pi$​求期望所必须的。所以我们通过一个等价变换来解决这个问题。

![67](https://pic.imgdb.cn/item/60fbba5f5132923bf884c56d.png)

这个量可以通过每步的采样计算得到，使用随机梯度上升，我们就得到了REINFORCE算法的更新式：

![68](https://pic.imgdb.cn/item/60fbba5f5132923bf884c587.png)

这个更新量的方向是参数空间中使得将来在状态$S_t$下重复选择动作$A_t$的概率增加最大的方向，大小正比于回报，反比于选择动作的概率。前者的意义在于它使得参数向着更有利于产生最大回报的动作的方向更新。后者的意义在于如果不这样的话，频繁被选择的动作会占优（在这些方向更新更频繁），即使这些动作不是产生最大回报的动作。

![69](https://pic.imgdb.cn/item/60fbba5f5132923bf884c59e.png)

和算法流程中最后一行和之前有些不同。第一个不同是$\frac{\nabla\pi}{\pi}$变成了$\nabla \ln \pi$，这两者其实是等价的。在历史文献中，这个向量有许多不同的名字和记法，这里我们叫它<font color=red>迹向量（eligibility vector）</font>。

另一个不同是更新式中多了一个$\gamma^t$，这是考虑了折扣之后的结果。策略梯度定理中主要结论会有一些是适当的调整。

![70](https://pic.imgdb.cn/item/60fbba5f5132923bf884c5af.png)

## 带有基线的REINFORCE

![71](https://pic.imgdb.cn/item/60fbba8a5132923bf88579d5.png)

这个基线可以是任意的函数，甚至是一个随机变量，**只要不随动作$a$变化**，上式等式仍然成立，因为减的那一项等于0

![72](https://pic.imgdb.cn/item/60fbba8a5132923bf88579ec.png)

于是就有了包含基线的新的REINFORCE版本

![73](https://pic.imgdb.cn/item/60fbba8a5132923bf8857a07.png)

一般，加入这个基线不会使更新的期望发生变化，但是对方差会有很大的影响。在之前介绍的梯度赌博机算法中，一个类似的基线可以极大地降低方差（因而也加快了学习速度）。在赌博机中，基线就是一个数值（到目前为止的平均收益），但是对于马尔科夫决策过程，这个基线应该根据状态的变化而变化。比如，在一些状态下，所有动作的价值可能都比较大，因此我们需要一个较大的基线用以区分<font color=red>更大值的动作和相对值不那么高的动作</font>。

One natural choice for the baseline is an estimate of the state value $\hat{v}(S_t, w)$，which can also be learned by a Monte-Carlo method.

![74](https://pic.imgdb.cn/item/60fbba8a5132923bf8857a28.png)

This algorithm has two step sizes, denoted $\alpha^{\theta}$ and  $\alpha^w$. Choosing the step size for values (here $\alpha^w$) is relatively easy; in the linear case we have rules of thumb for setting it, such as $\alpha^w=0.1/\mathbb{E}[\parallel \nabla\hat{v}(S_t, w) \parallel^2]$. It is much less clear how to set the step size for the policy parameters, $\alpha^{\theta}$, whose best value depends on the range of variation of the rewards and on the policy parameterization.

## “行动器-评判器”方法



尽管带基线的强化学习方法既学习了一个策略函数也学习了一个状态价值函数，我们也不认为它是一种“Actor-Critic”方法，因为它的状态价值函数仅被用作基线，而不是作为“评判器”。也就是说，它没有被用于自举操作（用后继状态的价值估计来更新当前某个状态的价值估计值），而只是作为正被更新的状态价值的基线。

这一段的英文原文是这样的：

In REINFORCE with baseline, the learned state-value function estimates the value of the only the first state of each state transition. This estimate sets a baseline for the subsequent return, but is made **prior to the transition’s action and thus cannot be used to assess that action**. In actor–critic methods, on the other hand, the state-value function is applied also to the second state of the transition. <font color=red>The estimated value of the second state, when discounted and added to the reward, constitutes the one-step return, $G_{t:t+1}$, which is a useful estimate of the actual return and thus is a way of assessing the action</font>.

做这个区分是很有必要的，因为只有采用自举法时，才会出现依赖于函数逼近质量的偏差和渐近收敛性。

通过自举法引入的偏差以及状态表示上的依赖经常是很有用的，因为它们降低了方差并加快了学习。带基线的强化学习方法是无偏的，并且会渐近地收敛至局部最小值，但是和所有蒙特卡洛方法一样，它的学习比较缓慢（产生高方差估计）。

![75](https://pic.imgdb.cn/item/60fbba8a5132923bf8857a3d.png)

![76](https://pic.imgdb.cn/item/60fbbae45132923bf886f2ac.png)

## 持续性问题的策略梯度

对于没有分幕边界的持续性问题，我们需要根据每个时刻上的平均收益来定义性能

![77](https://pic.imgdb.cn/item/60fbbae45132923bf886f2bf.png)

其中，$\mu$​是策略$\pi$​下的稳定状态的分布，$\mu(s) \doteq \lim_{t \to \infty}Pr(S_t=s\|A_{0:t} \sim \pi)$​，并假设它一定存在并独立于$S_0$​（一种遍历性假设）。注意，这是一个很特殊的状态分布，如果一直根据策略$\pi$​选择动作，则这个分布会保持不变：

![78](https://pic.imgdb.cn/item/60fbbae45132923bf886f2d5.png)

在持续性问题中，我们用**差分回报**定义价值函数，

![79](https://pic.imgdb.cn/item/60fbbae45132923bf886f2f8.png)

有了这些定义，适用于分幕情况下的策略梯度定理对于持续性任务的情况也同样正确。

![80](https://pic.imgdb.cn/item/60fbbae45132923bf886f315.png)

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
