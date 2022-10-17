---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter4
subtitle:   Dynamic Programming
date:       2021-04-15
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

利用贝尔曼方程，使用迭代法来解方程组。考虑一个近似的价值函数序列，$v_0,v_1,\ldots,$，从$S^+$映射到$R$（实数集）。初始的近似值$v_0$可以任意选取（<font color=red> 除了终止状态值必须为0外</font>）。然后下一轮迭代使用$v_{\pi}$的贝尔曼方程进行更新。

![18](https://pic.imgdb.cn/item/60fbb1b45132923bf8601084.png)

在写算法的时候，我们可以使用两个数组：一个用于存储旧的价值函数$v_k(s)$，一个用于存储新的价值函数$v_{k+1}(s)$。也可以简单地使用一个数组来进行“就地”（in place）更新。这里有一个小点需要注意，就是根据状态更新的顺序，上式的右端有时候会使用新的价值函数。就地更新的算法反而会比传统的更新收敛得更快。

一般来说，一次更新是对整个状态空间的一次**遍历**（sweep）。<font color=red> 对于就地更新算法，遍历的顺序对于收敛的速率有着很大的影响</font>。

## 策略改进

假设对于任意一个确定的策略$\pi$，我们已经知道了它的价值函数$v_{\pi}$。对于某个状态$s$，我们想知道是否应该选择一个不同于给定策略的动作$a\neq \pi(s)$。我们知道，如果从状态$s$继续使用现有的策略，那么最后的结果就是$v_{\pi}(s)$。但是如果使用一个新策略，结果是会更好还是更坏？一种解决方法是在状态$s$选择一个动作$a$，然后继续遵循现有的策略$\pi$。这样得到的结果是：

![19](https://pic.imgdb.cn/item/60fbb1b45132923bf860109f.png)

所以关键就是这个值是大于还是小于$v_{\pi}(s)$。如果这个值更大，这说明新策略（在状态$s$选择动作$a$，然后继续使用策略$\pi$）会比始终使用策略$\pi$更优。

基于这个想法，我们可以根据原策略的价值函数执行贪心算法，来构造一个更好的策略。

![20](https://pic.imgdb.cn/item/60fbb1b45132923bf86010b6.png)

## 网格世界（grid world）

![21](https://pic.imgdb.cn/item/60fbb31e5132923bf865ea4d.png)

![26](https://pic.imgdb.cn/item/60fbb4275132923bf86a5aa3.png)

### 代码

所有这本书中的例子的代码我都整理到到[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)这个github上了。

## 杰克租车问题

![22](https://pic.imgdb.cn/item/60fbb31e5132923bf865ea60.png)

### 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。

## 策略迭代（Policy Iteration）和价值迭代（Value Iteration）

**策略迭代**的思路很朴素，就是策略评估（policy evaluation）+ 策略改进（Policy Improvement）。但是在这个过程中策略评估本身也是一个需要多次遍历状态集合的迭代过程，那么收敛到$v_{\pi}$也只是理论上在极限处才成立（这里要注意，不是说策略评估是一个迭代过程，而是说如果使用迭代的方法来做策略评估，策略评估不止一种方法，比如直接解线性方程组就能得到精确解）。<font color=red> 我们是否必须等到其完全收敛</font>，还是可以提早结束？上面的grid world的例子说明是可以的。

事实上，有多种方式可以截断策略迭代中的策略评估步骤，并不影响其收敛。一种重要的特殊情况是，在一次遍历后即刻停止策略评估（对每个状态进行一次更新）。该算法就被称为**价值迭代**。

另一种理解价值迭代的方法就是通过贝尔曼最优方程。价值迭代就是将贝尔曼最优方程作为了更新规则。

## 赌徒问题

![23](https://pic.imgdb.cn/item/60fbb31e5132923bf865ea7a.png)

这里有个点要注意一下，就是action的取值范围，上图中说的是，如果gambler手中有赌资$s$，那么他下注的值是从0到$\min(s, 100-s)$，并不是$[0,s]$。这也就是说，<font color=red> 当gambler手上有超过50时，他永远不会拿全部的赌资下注</font>。

为什么赌徒问题的最优策略是一个如此奇怪的形式？

首先，因为硬币正面朝上的概率是小于0.5的，也就是对赌徒来说是不利的。因此，赌徒应该竟可能地去减少扔硬币的次数。在50的时候赌徒可以赢的概率是0.4。在51的时候，此时应该将其分成50和1来考虑（我也没理解为啥？），如果用1来下注，赢了的话可以变成52，即使输了也就是退回到50，依然有0.4的概率赢。

然后，如果你写了做了书上的习题，就会发现当硬币正面朝上的概率$\geq 0.5$时，其实value iteration要经过好多次才收敛，并且长得已经完全不一样了，如下图

![27](https://pic.imgdb.cn/item/60fbb4275132923bf86a5ac4.png)

你会发现这时的最优策略已经改变了：不管赌徒手上有多少赌资，他永远只会拿出1去下注。理由的话，自己的理解，因为每扔1次都是对自己有利的，所以创造尽可能多扔的情况就是最优的选择。

这个问题这么想下来其实还蛮有意思的。我还找到了一篇专门讲如何赌博的文章[How to Gamble If You Must](https://www.maa.org/sites/default/files/pdf/joma/Volume8/Siegrist/RedBlack.pdf)，感兴趣的可以去看看，哈哈哈。

### 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。

## 异步动态规划（Asynchronous Dynamic Programming）

动态规划的一个主要缺点是：需要对整个状态集进行多次遍历。如果状态集很大，那么即使单次遍历也会十分昂贵。

异步DP算法是一类就地迭代的DP算法，其不以系统遍历状态集的形式来组织算法。这些算法使用任意可用的状态值，以任意的顺序来更新状态值。在某些状态的值更新一次之前，另一些状态的值可能已经更新了好几次。然而，为了正确收敛，<font color=red> 异步算法必须不断地更新所有的状态值：在某个计算节点后，它不能忽略任何的状态</font>。

**当然，避免遍历并不一定意味着我们可以减少计算量。这只是意味着，一个算法在改进策略前，不需要陷入任何漫长而无望的遍历。**

我们也可以试着调整更新的顺序。使得价值信息能更有效地在状态间传播。对于一些状态可能不需要像其他状态那样频繁地更新其价值函数。如果它们与最优行为无关，我们甚至可以完全跳过一些状态。

异步算法还使计算和实时交互的结合变得更加容易。我们可以在智能体实际在MDP中进行真实交互的同时，执行DP算法。<font color=red> 智能体的经历可以用于确定DP算法要更新哪个状态。与此同时，DP算法的最新值和策略信息可以指导智能体的决策</font>。比如，我们可以在智能体访问状态时将其更新。这使得将DP算法的更新**聚焦**到部分与智能体最相关的状态集成为可能。

## 广义策略迭代（Generalized Policy Iteration）

我们用**广义策略迭代**（GPI）一词来指代让策略评估和策略改进相互作用的一般思路，与这两个流程的粒度和其他细节无关。

![24](https://pic.imgdb.cn/item/60fbb31e5132923bf865ea90.png)

几乎所有的强化学习方法都可以被描述成GPI。策略总是基于特定的价值函数进行改进，价值函数也始终会向对应特定策略的真实价值函数收敛。价值函数只有在与当前策略一致时才稳定，并且策略只有在对当前价值函数是贪心策略时才稳定。因此，<font color=red> 只有当一个策略发现它对自己的评估价值函数是贪心策略时，这两个流程才能稳定下来</font>。这意味着贝尔曼最优方程成立。

![25](https://pic.imgdb.cn/item/60fbb31e5132923bf865eaa9.png)

我们可以将GPI的评估和改进流程视为两个约束或者目标之间的相互作用的流程。每个流程都把价值函数或策略推向其中的一条线，改线代表了某一个目标的解决方案。直接冲向一个目标会导致某种程度上地偏离另一个目标。然而，整体流程的结果会更接近最优的总目标。在GPI中，我们也可以让每次走的步子小一点，部分地实现其中一个目标。

DP算法有时会由于**维度灾难**而被认为缺乏实用性。维度灾难是指，状态的总数量经常随着状态变量的增加而指数级上升。状态集合太大的确会带来一些困难，但是这些是问题本身的苦难，而非DP作为一种揭发所带来的困难。事实上，相比于直接搜索和线性规划，DP找到最优化策略的时间（最坏情况下）与状态和动作的数量是呈多项式关系的，反而更适合于解决大规模状态空间问题。并且我们还有异步DP这个选项。

DP算法有一个特殊的性质。所有的方法都是根据对后继状态价值的估计来更新对当前状态价值的估计。基于其他估计来更新自己的估计，这个普遍的思想被称为**自举法**。（That is, they update estimates on the basis of other estimates. We call this general idea bootstrapping.）