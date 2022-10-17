---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter10
subtitle:   On-policy Control with Approximation
date:       2021-06-27
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

本章以半梯度Sarsa算法为例，它是上一章讨论的半梯度TD(0)算法到动作价值，以及到策略控制的自然延伸。在分幕式任务的情况下，这种延伸是直接的，但是在持续性任务的情况下，我们必须退回几步来重新审视如何使用折扣来定义最优策略。令人惊讶的是，<font color=red>一旦我们有了真正的函数逼近，我们就必须放弃折扣并将控制问题的定义转换为一个新的“平均收益”的形式，这个形式有一种新的“差分”价值函数</font>。

## 分幕式半梯度控制

![177](https://pic.imgdb.cn/item/60fbc7e25132923bf8bf20ec.png)

我们把这种方法称为*分幕式半梯度单步Sarsa*。对于一个固定策略，这个方法的收敛情况与TD(0)一样，具有相同的误差边界。

To form control methods, we need to couple such action-value prediction methods with techniques for policy improvement and action selection. 通过将估计策略变为贪心策略的一个柔性近似，如$\epsilon$-贪心策略，策略改进便完成了。也是用同样的策略来选取动作。

![178](https://pic.imgdb.cn/item/60fbc7e25132923bf8bf2101.png)

### Example 10.1: 高山行车（Mountain Car）问题

该问题的困难在于重力比汽车发动机更强，即使在全油门下汽车也不能驶上陡坡。The only solution is to first move away from
the goal and up the opposite slope on the left. Then, by applying full throttle the car can build up enough inertia to carry it up the steep slope even though it is slowing down the whole way.  **在某种意义上，事情会先变得更糟（离目标越远），然后才变得更好。**

![179](https://pic.imgdb.cn/item/60fbc7e25132923bf8bf2113.png)

![180](https://pic.imgdb.cn/item/60fbc7e25132923bf8bf2126.png)

以上结果有几个地方需要注意一下：

1. 这个任务中所有的真实价值都是负数，而最初的动作价值函数都是零，这是乐观的，这使得即使试探参数$\epsilon$为0，也会引起广泛的试探。
2. 我们在这里不使用蒙特卡洛方法，是因为该问题可能会出现一幕过长，甚至永远不结束的情况，从而导致训练无法进行。
3. 图10.4中较大的n比较小的n有更高的标准差。原因是n-步方法是必须等到n步之后才能更新，而且辐射到的状态也会越多。在初期agent会有一些很差的action，n越大这些差的估计就会传到更多的状态，导致一些好的状态的初始估计离真实值差距很大，从而导致方差变大。

## 平均收益：持续性任务中的新的问题设定

The average-reward setting is one of the major settings commonly considered in the classical theory of dynamic programming and less-commonly in reinforcement learning. The discounted setting is problematic with function approximation, and thus the average-reward setting is needed to replace it.

![181](https://pic.imgdb.cn/item/60fbc97d5132923bf8c5ca07.png)

In particular, we consider all policies that attain the maximal value of $r(\pi)$ to be optimal.

![182](https://pic.imgdb.cn/item/60fbc97d5132923bf8c5ca14.png)

![183](https://pic.imgdb.cn/item/60fbc97d5132923bf8c5ca25.png)

![184](https://pic.imgdb.cn/item/60fbc97d5132923bf8c5ca3d.png)

上述伪代码使用$\delta_t$​​​​​​作为误差来更新$\bar{R}\_{t+1}$​​​​​​，而不是简单地用$R_{t+1}-\bar{R}\_t$​​​​​​来更新。虽然这两个误差定义都可以，但是$\delta_t$​​​​​​的效果更好。为了探明原因，我们讨论练习10.7中三个状态的环状MRP。平均收益的估计应该趋向于它的真实值$\frac{1}{3}$​​​​​​。**假设已经达到这个值并保持固定**。误差$R_{t+1}-\bar{R}\_t$​​​​​​的序列是$-\frac{1}{3},-\frac{1}{3},\frac{2}{3},-\frac{1}{3},-\frac{1}{3},\frac{2}{3}\cdots$​​​​​​，而$\delta_t$​​​​​​的序列为$0,0,0,\cdots$​​​​​​。因此，使用$\delta_t$​​​​​​会产生更为稳定的估计，虽然知道是后面的$\hat{V}(S\_{t+1})-\hat{V}(S\_t)$​​​​​​起作用的，但是具体原因没明白。

### Example 10.2: 访问控制队列（Access-Control Queuing）任务

![185](https://pic.imgdb.cn/item/60fbc97d5132923bf8c5ca52.png)

## 弃用折扣

The continuing, discounted problem formulation has been very useful in the tabular case, in which the returns from each state can be separately identified and averaged. But in the approximate case it is questionable whether one should ever use this problem formulation.

为了搞清楚为什么，我们考虑一个没有开始或结束的无限长的收益序列，也没有清晰定义的可区分的状态。这些状态可能仅仅由特征向量来表示，而它们对于区分不同的状态可能作用不大。作为一个特例，所有特征向量可能都是一样的。Thus one really has only the reward sequence (and the actions), and performance has to be assessed purely from these. How could it be done?一种方法是通过计算较长时间间隔的收益的平均来进行性能评估——这就是平均收益的设定。那么如何使用折扣呢？对于每一个timestep我们可以计算折后回报。有些回报很大，有些很小，所以我们仍然需要在足够大的时间间隔中进行平均。在持续性问题上没有开始和结束，没有特殊的时刻。事实上，对于策略$\pi$，折后回报的均值总是$r(\pi)/(1-\gamma)$，也就是说，它本质上就是平均收益。In particular, the ordering of all policies in the average discounted return setting would be exactly the same as in the average-reward setting.

下面的方框里给出了证明，其基本思想可以通过对称的观点来解释。每一个时刻都和其他时刻一样。有折扣的情况下，每一个收益都会在回报的某个位置恰好出现一次。第$t$个收益会无折扣地出现在第$t-1$个回报中，在第$t-2$个汇报中打一次折扣，在第$t-1000$个回报中打999次折扣。所以第$t$个收益的权重就是$1+\gamma+\gamma^2+\cdots=1/(1-\gamma)$。因为所有状态都是一样的，它们都是这个权重，所以折后回报的平均值就是这个系数乘以平均收益。

![186](https://pic.imgdb.cn/item/60fbca3d5132923bf8c8df35.png)

这表明折扣在连续型任务的定义中不起作用。但是我们仍然可以将折扣用于解决方案中。Unfortunately, **discounting algorithms with function approximation** do not optimize discounted value over the on-policy distribution, and thus are not guaranteed to optimize average reward（在表格型下是可以的）.

<font color=red>使用函数逼近的折扣控制设定困难的根本原因在于我们失去了策略改进定理。我们在单个状态上改进折后价值函数不再保证我们会改进整个策略。</font>而这个保证是强化学习控制方法的核心，使用函数逼近让我们失去了这个核心！



事实上，策略改进定理的缺失也是分幕式设定以及平均收益设定的理论缺陷。一旦引入了函数逼近，我们无法保证在任何设定下都一定会有策略的改进。In Chapter 13 we introduce an alternative class of reinforcement learning algorithms based on parameterized policies, and there we have a theoretical guarantee called the “policy-gradient theorem” which plays a similar role as the policy improvement theorem. 但是对于学习动作价值的方法，还没有一个局部的改进保证（possibly the approach taken by Perkins and Precup (2003) may provide a part of the answer）。我们确实知道的是，$\epsilon$-贪心优化有时会导致一个较差的策略，因为算法可能在若干好的策略之间来回摆动而不收敛到其中的一个（Gordon，1996）。

## 差分半梯度n步Sarsa

![187](https://pic.imgdb.cn/item/60fbca3d5132923bf8c8df48.png)

在上述算法中，平均收益的步长参数$\beta$​需要足够小，以保证$\bar{R}$​是一个较好的对平均收益的长期估计。Unfortunately, $\bar{R}$​ will then be biased by its initial value for many steps, which may make learning inefficient.  因而可以采用采样观测的收益的平均值来代替$\bar{R}$​。其会在前期迅速适应，但长期来看仍然适应得很慢。而且由于策略缓慢的变化，$\bar{R}$​也会发生变化。这种可能存在的长期非平稳性使得采样平均法并不适用。事实上，**平均收益的步长参数**是使用练习2.7中无偏常数不长的技巧的最佳之地。

![188](https://pic.imgdb.cn/item/60fbca3d5132923bf8c8df5a.png)

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
