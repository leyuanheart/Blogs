---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter7
subtitle:   n-step Bootstrapping
date:       2021-07-08
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

In many applications one wants to be able to update the action very fast to take into account anything that has changed, but bootstrapping works best if it is over a length of time in which **a significant and recognizable state change has occurred**.

n步时序差分方法的思想一般会作为资格迹（eligibility trace）算法思想的一个引子，资格迹算法允许自举法在多个时间段同时开展操作。

*n-step return*

![93](https://pic.imgdb.cn/item/60fbbb6a5132923bf8892522.png)

注意，计算n步回报（$n>1$）需要涉及若干将来时刻的收益和状态，当前状态从当前时刻t转移到下一时刻t+1时，我们没法获取这些收益和状态。n步回报只有在获取到$R_{t+n}$和$V_{t+n-1}$后才能计算，而这些值都只能在时刻$t+n$时才能得到。The natural state-value learning algorithm for using n-step returns is thus

![94](https://pic.imgdb.cn/item/60fbbb6a5132923bf8892534.png)

<font color=red> 注意，在最开始的n-1个时刻，价值函数不会被更新。为了弥补这个缺失，在终止时刻后还将执行对应次数的更新。</font>

![95](https://pic.imgdb.cn/item/60fbbb6a5132923bf8892556.png)

![96](https://pic.imgdb.cn/item/60fbbef65132923bf898bf7b.png)

![97](https://pic.imgdb.cn/item/60fbbef65132923bf898bf8b.jpg)

n步回报利用价值函数$V_{t+n-1}$来校正$R_{t+n}$之后的所有剩余收益之和。n 步回报的一个重要特性是：在最坏的情况下，采用它的期望值作为对$v_{\pi}$的估计可以保证比$V_{t+n-1}$更好。换句话说，n步回报的期望的最坏误差能够保证不大于$V_{t+n-1}$最坏误差的$\gamma^n$倍

![98](https://pic.imgdb.cn/item/60fbbef65132923bf898bfa1.png)

此式对于任意的$n\geq1$都成立。这被称为n步回报的**误差减小**性质。根据这个性质，可以证明所有的n步时序差分方法在合适的条件下都能收敛到正确的预测。

## n步时序差分方法在随机游走上的应用

![99](https://pic.imgdb.cn/item/60fbbef65132923bf898bfb9.png)

小任务会更倾向于用小的n，<font color=red>因为当$n \geq \frac{\# state - 1}{2}$，所有被访问的状态都可以通过n步到达terminal，这样就不存在bootstrap，完全是MC更新</font>。（表示怀疑）

而reward变成-1也会有利于小的n，因为在比较长的轨迹中，大的n意味着很多在n步之内到达终止状态的状态会被更新（这些更新是没有利用bootstrap的，因为终止状态的价值为0）,这会导致方差变大。

# n步Sarsa

The main idea is to simply switch states for actions (state–action pairs) and then use an "$\epsilon$-greedy policy.

![102](https://pic.imgdb.cn/item/60fbbfad5132923bf89bf0fb.png)

![100](https://pic.imgdb.cn/item/60fbbef65132923bf898bfcf.jpg)

关于为什么n步Sarsa能够加速学习，参见下图。

![103](https://pic.imgdb.cn/item/60fbbfad5132923bf89bf113.png)

![101](https://pic.imgdb.cn/item/60fbbfad5132923bf89bf0e3.png)

n步期望Sarsa和n步Sarsa的不同之处在于它的最后一个节点是一个分支，其对所有可能的动作采用策略$\pi$进行了加权。

![104](https://pic.imgdb.cn/item/60fbbfad5132923bf89bf129.png)

<font color=red>如果$s$是终止状态，那么我们定义它的期望估计价值为0。</font>

# n步离轨策略学习

Recall that off-policy learning is learning the value function for one policy, $\pi$, while following another policy, $b$. Often, $\pi$ is the greedy policy for the current action-value function estimate, and b is a more exploratory policy, perhaps "$\epsilon$-greedy. In order to use the data from $b$ we must take into account the difference between the two policies, using their relative probability of taking the actions that were taken.

在n步方法中，我们感兴趣的是这n步的相对概率。例如，实现一个简单离轨策略版本的n步时序差分学习，对于t时刻的更新（实际上更新发生在$t+1$时刻），可以简单地用$\rho_{t:t+n}$来加权：

![105](https://pic.imgdb.cn/item/60fbbfad5132923bf89bf148.png)

![106](https://pic.imgdb.cn/item/60fbc0005132923bf89d65f0.png)

For example, if any one of the actions would never be taken by $\pi$​ (i.e., $\pi(A_k\|S_k) = 0$​) then the n-step return should be given zero weight and be totally ignored. On the other hand, if by chance an action is taken that $\pi$​ would take with much greater probability than $b$​ does, then this will increase the weight that would otherwise be given to the return. <font color=red>This makes sense because that action is characteristic of $\pi$​ (and therefore we want to learn about it) but is selected only rarely by b and thus rarely appears in the data. To make up for this we have to over-weight it when it does occur</font>.

之前的n步Sarsa的更新方法可以完整地被如下简单的离轨策略版方法代替。对于$0\leq t < T$,

![107](https://pic.imgdb.cn/item/60fbc0005132923bf89d65fd.png)

注意，这里的重要度采样比，其起点和终点比n步时序差分学习都要晚一步。这是因为在这里我们更新的是“状态-动作”二元组。这时我们并不需要关心这些动作有多大概率被选择，既然我们已经确定了这个动作，那么我们想要的是充分地学习发生的事情，这个学习过程会使用基于后继动作计算出来的重要度采样加权系数。

离轨策略版本的n步期望Sarsa使用和上面的更新类似，除了重要度采样率使用$\rho_{t+1:t+n-1}$，以及使用期望Sarsa版本的n步回报。这是因为在期望Sarsa中，所有可能的动作都会在最后一个状态中被考虑，实际采取的那个具体动作对期望Sarsa的计算没有影响，也就不需要去修正。

# *带控制变量的每次决策型方法

这一节及之后的内容我感觉挺绕的，但是我也觉得是有一些涉及强化学习核心的思想的。我先按照自己的理解梳理一遍，不过我应该需要再多看几遍才能完全理解。

上面介绍的多步离轨策略方法非常简单，在概念上很清晰，但是可能不是最高效的。一种更精巧、复杂的方法采用了蒙特卡洛方法那一章里介绍的“每次决策型重要度采样”思想。

为了理解这种方法，首先要注意到普通的n步方法的回报，像所有的回报一样，可以写成递归的形式。对于视界终点为h的n个时刻，n步回报可以写成

![108](https://pic.imgdb.cn/item/60fbc0005132923bf89d661e.png)

其中$G_{h:h} \doteq V_{h-1}(S_h)$​。现在考虑遵循不同于目标策略$\pi$​的行动策略$b$​而产生的影响。所有产生的经验，包括对于t时刻的第一次收益$R_{t+1}$​和下一个状态$S_{t+1}$​都必须用重要度采样率$\rho_t=\frac{\pi(A_t\|S_t)}{b(A_t\|S_t)}$​加权。也许有人觉得，直接对上面等式右侧加权不就行了吗？但其实我们有更好的方法，假设t时刻的动作永远都不会被$\pi$​选择，则$\rho_t$​为0。那么简单加权会让n步方法的回报为0，当它被当做目标时可能会产生很大的方差。而更精巧的方法中，我们采用一个替换定义。在视界h结束的n步回报的离轨策略版本的定义可以写成

![109](https://pic.imgdb.cn/item/60fbc0005132923bf89d6641.png)

在这个方法中，如果$\rho_t=0$，则它并不会使目标为0并导致估计值收缩，而是使得姆目标和估计值一样，因此不会带来任何变化（这是合理的）。上式中第二项额外添加的项被称为**控制变量**。注意这个控制变量并不会改变更新值的期望：<font color=red>重要度采样率的期望为1并且与估计值没有关联</font>。

上式是

![93](https://pic.imgdb.cn/item/60fbbb6a5132923bf8892522.png)

的严格推广。当$\rho_t$恒为1时，这两个式子完全等价。

有了这个新的$G_{t:h}$的定义，我们就可以采用

![94](https://pic.imgdb.cn/item/60fbbb6a5132923bf8892534.png)

来进行更新（它没有明显的重要度采样率，除了嵌入在回报当中的部分）。

对于动作价值，n步的离轨策略版本定义稍有不同。**主要是因为第一个动作在重要度采样中不起作用**。

首先，对于动作价值而言，n步回报也可以写成一样的递归形式，而使用控制变量的离轨策略的形式**可以为**

![110](https://pic.imgdb.cn/item/60fbc0005132923bf89d665b.png)

如果$h<T$​，则上面的递归公式结束于$G_{h:h}=Q_{h-1}(S_h, A_h)$​（如果是期望Sarsa，则为$\bar{V}_{h-1}(S_h)=\sum_a \pi(a\|S_h)Q_{h-1}(S_h, a)$​）；如果$h\geq T$​，则递归公式结束于$G_{T-1:h}=R_T$​。由此结合

![111](https://pic.imgdb.cn/item/60fbc06d5132923bf89f5424.png)

可以得到期望Sarsa的n步离轨策略预测算法的扩展。

![112](https://pic.imgdb.cn/item/60fbc06d5132923bf89f5437.png)

![113](https://pic.imgdb.cn/item/60fbc06d5132923bf89f544d.png)

The importance sampling that we have used in this section, the previous section, and in Chapter 5, enables sound o↵-policy learning, but also results in high variance updates, forcing the use of a small step-size parameter and thereby causing learning to be slow. **It is probably inevitable that off-policy training is slower than on-policy training—after all, the data is less relevant to what is being learned.**

# 不需要使用重要度采样的离轨策略学习方法：n步树回溯（Tree Backup）算法

不使用重要度采的离轨策略学习算法可能存在吗？对于单步的情况，之前介绍的Q-learning和期望Sarsa就是的，但是有没有多步的算法呢？接下来要介绍的**树回溯**算法就是。

<font color=red>所有不使用重要度采样的算法的核心都是用来更新的量和是否采用目标算法无关，要么是求期望求掉了，要么是假设其就是来自目标分布（单步情况下的Q-learning）。</font>

![114](https://pic.imgdb.cn/item/60fbc06d5132923bf89f5465.png)

更精确地说，更新量是从树的叶子节点的动作价值的估计值计算出来的。内部的动作节点（对应于实际采取的动作），并不参与回溯。每个叶子节点的贡献会被加权，权值与它在目标策略$\pi$​​​下出现的概率成正比。因此除了被实际被采用的动作$A_{t+1}$​​​完全不产生贡献外，一个一级动作a的贡献权值为$\pi(a\|S_{t+1})$​​​。它的$\pi(A_{t+1}\|S_{t+1})$​​​被用来给所有二级动作的价值加权。所以每个未被选择的二级动作$a'$​​​的贡献权值为$\pi(A_{t+1}\|S_{t+1})\pi(a'\|S_{t+2})$​​​，以此类推。

树回溯算法的单步回报与期望Sarsa相同，对于$t<T-1$，有

![115](https://pic.imgdb.cn/item/60fbc06d5132923bf89f5473.png)

对于$t<T-2$，两步树回溯的回报是

![116](https://pic.imgdb.cn/item/60fbc0d35132923bf8a11860.png)

这显示树回溯的n步回报的递归定义的一般形式。对于$t<T-1, n \geq 2$，有

![117](https://pic.imgdb.cn/item/60fbc0d35132923bf8a11875.png)

除了$G_{T-1:t+n}=R_T$之外，$n=1$的情况可以由单步回报解决。

![118](https://pic.imgdb.cn/item/60fbc0d35132923bf8a11898.png)

![119](https://pic.imgdb.cn/item/60fbc0d35132923bf8a118be.png)

![120](https://pic.imgdb.cn/item/60fbc0d35132923bf8a118da.jpg)

# *一个统一的算法：n步$Q(\sigma)$

到目前为止，我们已经考虑了3种不同的动作价值回溯算法。n步Sarsa中的节点转移全部基于采样得到的单步路径，而树回溯算法对于状态到动作的转移，则是将所有可能路径分支全部展开而没有任何采样，n步期望Sarsa算法则只对最后一个状态到动作的转移进行全部分支展开（用期望值），其余转移则都是基于采样。这些算法在多大程度上可以统一？

![121](https://pic.imgdb.cn/item/60fbc1165132923bf8a23949.png)

一种统一框架的思想如上图第4个回溯图。这个思想就是：**对状态逐个决定是否要采取采样操作**。我们可以考虑采样和期望之间的连续变化。令$\sigma_t \in [0, 1]$表示在步骤$t$时的采样程度，$\sigma=1$表示完整的采样，$\sigma=0$表示求期望而不进行采样。随机变量$\sigma_t$可以表示成在时间$t$时，关于状态、动作或者“状态-动作”二元组的函数。

现在我们来推导n步$Q(\sigma)$的公式。首先我们写出视界$h=t+n$时的树回溯算法的n步回报

![122](https://pic.imgdb.cn/item/60fbc1165132923bf8a2395b.png)

经过上面的重写，这个式子的形式就很像带控制变量的n步Sarsa的回报，不同之处是用动作概率$\pi(A_{t+1}\|S_{t+1})$​代替了重要度采样比$\rho_{t+1}$​。对于$Q(\sigma)$​，我们会将两种线性情况组合起来，即对$t<h \leq T$​，有

![123](https://pic.imgdb.cn/item/60fbc1165132923bf8a23973.png)

上面递归公式的终止条件为：$h<T$时$G_{h:h} \doteq Q_{h-1}(S_h,A_h)$，或者$h=T$时$G_{T-1:h} \doteq R_T$。

![124](https://pic.imgdb.cn/item/60fbc1165132923bf8a2398a.png)

## Summary

所有n步方法都要在更新之前延迟n个时刻步长，因为这些方法必须知道一些未来发生的事情才能进行计算，另一个缺点是，它们涉及的计算量比之前的方法要大。与单步方法相比，n步方法还需要更多的内存来记忆最近的n步中的状态、动作、收益以及其他变量。

Eventually, in Chapter 12, we will see how multi-step TD methods can be implemented with minimal memory and computational complexity using eligibility traces.

尽管n步方法比使用资格迹的方法复杂很多，但在概念上它更清楚一些。

![125](https://pic.imgdb.cn/item/60fbc1165132923bf8a239aa.png)

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
