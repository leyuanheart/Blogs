---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter6
subtitle:   Temporal Difference Learning
date:       2021-05-11
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

![54](https://pic.imgdb.cn/item/60fbb7cc5132923bf879eb5f.png)

MC的目标之所以是一个“估计值”，是因为公式（6.3）中的期望值是未知的，我们用样本回报来代替期望回报。

DP的目标之所以是一个“估计值”则不是因为期望值的原因，其会假设由环境模型完整地提供期望值，真正的原因是因为真实的$v_{\pi}(S_{t+1})$是未知的，因此要使用当前的估计值$V(S_{t+1})$来替代。

TD的目标也是一个“估计值”。理由有两个：它采样得到对式（6.4）的期望值，并且使用当前的估计值$V$来代替真实值$v_{\pi}$。

![55](https://pic.imgdb.cn/item/60fbb7cc5132923bf879eb73.png)

在$TD(0)$的更新中，括号里的数值是一种误差，它衡量的是$S_t$的估计值与更好的估计$R_{t+1}+\gamma V(S_{t+1})$之间的差异。这个数值，被称为**TD误差**：

![56](https://pic.imgdb.cn/item/60fbb9c45132923bf8823195.png)

由于TD误差取决于下一个状态和一下个收益，所以<font color=red> $V(S_t)$中的误差$\delta_t$在$t+1$时刻才可获得</font>。还要注意，如果价值函数数组V在一幕内没有改变（例如，蒙特卡洛方法就是如此），则蒙特卡洛误差可以写为TD误差之和

![57](https://pic.imgdb.cn/item/60fbb9c45132923bf88231ad.png)

This identity is not exact if V is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately. 这个等式的泛化在TD学习的理论和算法中很重要。

如果V在幕中发生了变化，上式只能近似成立，等式两侧差在哪呢？

![58](https://pic.imgdb.cn/item/60fbb9c45132923bf88231c4.jpg)

## 时序差分的优势

Certainly it is convenient to learn one guess from the next, without waiting for an actual outcome, but can we still guarantee convergence to the correct answer?

幸运的是，答案是肯定的。对于任何固定的策略$\pi$，TD(0)都已经被证明能够收敛到$v_{\pi}$。如果步长参数是一个足够小的常数，那么它的均值能收敛到$v_{\pi}$。如果步长参数根据随机近似条件（求和等于无穷，平方求和小于有界）逐渐变小，则它能以概率1收敛。Most convergence proofs apply only to the table-based case of the algorithm, but some also apply to the case of general linear function approximation.

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is “Which gets there first?” In other words, which method learns faster? Which makes the more efficient use of limited data? At the current time this is an open question in the sense that no one has been able to prove mathematically that one method converges faster than the other. 不过在实践过程中，TD方法在随机任务上通常比常量$\alpha$MC方法收敛地更快。

### 例子：随机游走

![81](https://pic.imgdb.cn/item/60fbbb155132923bf887bd4a.png)

关于这个例子中true value的计算，因为知道每个状态的转移概率和对应的收益，是可以通过解线性方程组来得到的。但是还有一种更简单直观的解法，关键就是快速得到C点价值：由于对称性，从C点出发，一定是以0.5的概率到达左边，0.5的概率到达右边，所以很自然的，$V(C)=0.5$。有了$V(C)$，接下来就按照一定的顺序来算其他状态价值，比如，第二个要计算的是$V(E)$。因为$V(E)=0.5 + 0.5 * V(D)$，而$V(D)=0.5 * V(E) + 0.5 * V(C)$，联立两式，就能得到$V(E)=\frac{5}{6}$。以此类推。

还有一个需要注意的地方，就是这个例子的右图中，TD方法的MSE先下降后上升，尤其是在$\alpha$大时更明显。这种情况并不是总会发生，而是和这个例子中价值函数的初始化相关：$V(C)$恰好被初始化成了true value。在更新的早期，外围的状态会先更新，使得MSE逐渐减少，但是当TD error回传到了C，就使得其偏离了真实值，而且大的$\alpha$会加剧这种情况的发生。

## TD(0)的最优性

假设只有有限的经验，比如10幕数据或者100个时间步。在这种情况下，使用增量学习方法的一般方式是反复地呈现这些经验，直到方法最后收敛到一个答案为止。

给定近似的价值函数$V$，在访问非终止状态的每个时刻$t$，使用$\alpha (G_t - V(S_t))$或者$\alpha (R_{t+1} + V(S_{t+1}) - V(S_t))$计算相应的增量，但是价值函数仅根据所有增量的和改变一次。然后，利用新的值函数再次处理所有可用的经验，产生新的总增量，以此类推，直到价值函数收敛。我们称这种方法为**批量更新（batch updating）**。因为只有在处理了整批的训练数据后才进行更新。

Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter, $\alpha$, as long as $\alpha$ is chosen to be sufficiently small. The constant-$\alpha$ MC method also converges deterministically under the same conditions, but to a different answer. 

### 批量更新的随机游走

Batch-updating versions of TD(0) and constant-$\alpha$ MC were applied as follows to the random walk prediction example. After each new episode, all episodes seen so far were treated as a batch.
They were repeatedly presented to the algorithm, either TD(0) or constant-$\alpha$ MC, with $\alpha$ sufficiently small that the value function converged.

![82](https://pic.imgdb.cn/item/60fbbb155132923bf887bd60.png)

### 例子：你就是预测者

![83](https://pic.imgdb.cn/item/60fbbb155132923bf887bd7a.png)

每个人应该都会同意$V(B)$的最优估计是$\frac{3}{4}$，because six out of the eight times in state B the process terminated immediately with a return of 1, and the other two times in B the process terminated immediately with a return of 0.

但是在给定这些数据时，$V(A)$的最优估计是多少？下面有两个合理的答案。

![84](https://pic.imgdb.cn/item/60fbbb155132923bf887bd94.png)

这个答案的一种解读是认为它首先将其建模成上图的马尔科夫过程，再根据模型计算出正确的预测值。这是使用批量TD(0)方法会给出的结果。

另一种合理的答案是简单地观察到状态A只出现了一次并且对应的回报为0，因此就估计$V(A)=0$。这是批量蒙特卡洛方法会得到的答案。<font color=red> 注意这也是使得训练数据上平均误差最小的答案</font>。But still we expect the first answer to be better. If the process is Markov, we expect that the first answer will produce lower error on **future** data, even though the Monte Carlo answer is better on the existing data.

批量蒙特卡洛方法总是找出最小化训练集上均方误差的估计，而批量TD(0)方法总是找出完全符合马尔科夫过程模型的最大似然估计参数。Given this model, we can compute the estimate of the value function that would be exactly correct if the model were exactly correct. This is called the **certainty-equivalence estimate** because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated. In general, batch TD(0) converges to the certainty-equivalence estimate.

在以批量的形式学习的时候，TD(0)比蒙特卡洛方法更快是因为它计算的是真正的确定性等价估计。Although the non-batch methods do not achieve either the certainty-equivalence or the minimum squared-error estimates, they can be understood as moving roughly in these directions. Non-batch TD(0) may be faster than constant-$\alpha$ MC because it is moving toward a better estimate, even though it is not getting all the way there. **At the current time nothing more definite can be said about the relative efficiency of online TD and Monte Carlo methods.**

最后，值得注意的是，虽然确定性等价估计从某种角度上来看是一个最优答案，但是直接计算它几乎是不可能的。如果$n=\|S\|$​是状态数，那么仅仅建立过程的最大似然轨迹就可能需要$n^2$​的内存，如果按照传统方法，计算相应的价值函数则需要$n^3$​数量级的步骤。相比之下，TD方法则可以使用不超过$n$​的内存，并且通过在训练集上反复计算来逼近同样的答案，这一优势确实是惊人的。对于状态空间巨大的任务，TD方法可能是唯一可行的逼近确定性等价解的方法。

## Sarsa: 同轨策略下的时序差分控制

现在，我们使用时序差分方法来解决控制问题。按照惯例，我们仍然遵循广义策略迭代（GPI）的模式，只不过这次我们在评估和预测的部分使用时序差分方法。

The first step is to learn an action-value function rather than a state-value function.

The convergence properties of the Sarsa algorithm depend on the nature of the policy’s dependence on Q. For example, one could use $\epsilon$-greedy or $\epsilon$-soft policies. **Sarsa converges with probability 1 to an optimal policy and action-value function** as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with $\epsilon$-greedy policies by setting $\epsilon=\frac{1}{t}$)

### 例子：有风的网格世界

![85](https://pic.imgdb.cn/item/60fbbb155132923bf887bdaa.png)

## Q-learning: Off-policy TD Control

![86](https://pic.imgdb.cn/item/60fbbb3f5132923bf8886dcb.png)

In this case, <font color=red> the learned action-value function, Q, directly approximates $q_*$, the optimal action-value function, independent of the policy being followed</font>. This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs. **The policy still has an effect in that it determines which state–action pairs are visited and updated.** However, all that is required for correct convergence is that all pairs continue to be updated. This is a minimal requirement in the sense that any method guaranteed to find optimal behavior in the general case must require it. Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, Q has been shown to converge with probability 1 to $q_*$.

## Expected Sarsa

考虑一种与Q-learning十分类似但把对于下一个“状态-动作”二元组取最大值这一步换成取期望的学习算法。

![87](https://pic.imgdb.cn/item/60fbbb3f5132923bf8886dd9.png)

Given the next state, $S_{t+1}$, this algorithm moves deterministically in the same direction as *Sarsa* moves in expectation, and accordingly it is called *Expected Sarsa*.

**Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the random selection of $A_{t+1}$.**

![88](https://pic.imgdb.cn/item/60fbbb3f5132923bf8886dec.png)

在悬崖边行走这个例子中，状态的转移都是确定的，所有的随机性都来自于策略。在这种情况下，期望Sarsa可以放心地设定$\alpha=1$，而不要担心长期稳态性能的损失。与之相比，Sarsa仅仅在的值较小时能够有良好的长期表现，而启动期瞬态表现就很糟糕。In this and other examples there is a consistent empirical advantage of Expected Sarsa over Sarsa.

在悬崖边行走这个例子中，期望Sarsa被用作一种同轨策略的算法。但在一般情况下，它可以采用与目标策略$\pi$不同的策略来生成行为，这就变成了离轨策略的算法。For example, suppose $\pi$ is the greedy policy while
behavior is more exploratory; then Expected Sarsa is exactly Q-learning（对于贪心策略，取期望就等同于取max）. **In this sense Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa.** Except for the small additional computational cost, Expected Sarsa may completely dominate both of the other more-well-known TD control algorithms.

## Maximization Bias and Double Learning

**In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum value, which can lead to a significant positive bias.**

我们回顾一下Q-learning的target值：$y_t=r_t+\gamma \underset{a'}{\max}Q_w(s_{t+1},a')$

这个`max`操作会使Q value越来越大，甚至高于真实值，原因用数学公式来解释的话就是：对于两个随机变量$X_1$​和$X_2$​，有$E[\max(X_1,X_2)]\geq \max(E[X_1],E[X_2])$


$$
\begin{align}
\underset{a'}{\max}Q_w(s_{t+1},a') & = Q_w(s_{t+1}, \arg \underset{a'}{\max}Q_w(s_{t+1},a')) \\
& =E[G_t|s_{t+1}, \arg \underset{a'}{\max}Q_w(s_{t+1},a'),w] \\
& \geq max(E[G_t|s_{t+1},a_1,w],E[G_t|s_{t+1},a_2,w],\cdots) \\
& = max(Q_w(s_{t+1},a_1), Q_w(s_{t+1},a_2),\cdots)
\end{align}
$$


不等式右边才是我们真正想求的值，问题就出在选择action的网络和计算target Q value的网络是同一个网络。而且随着候选动作的增加，过高估计的程度也会越严重（可以理解成动作集的增加会增加随机性，更容易产生一个过高的估计）。

### 最大化偏差的例子

![89](https://pic.imgdb.cn/item/60fbbb3f5132923bf8886df8.png)

对于这个问题，有一种看法是，其根源在于确定价值最大的动作和估计它的价值这两个过程采用了同样的样本（多幕序列）。假如我们将这些样本划分为两个集合，并用它们学习两个独立的对真实价值$q(a)$​​​的估计$Q_1(a)$​​​和$Q_2(a)$​​​，那么我们接下来就可以使用其中一个估计，比$Q_1(a)$​​​，来确定最大的动作$A^\*= \arg \max_a Q_1(a)$​​​，再用另外一个$Q_2$​​​来计算其价值的估计$Q_2(A^\*)=Q_2(\arg \max Q_1(a))$​​​。由于$E[Q_2(A^\*)]=q(A^\*)$​​​，因此这个估计是无偏的。我们也可以交换两个估计的角色再执行一遍上面的过程，那么就又可以得到另一个无偏的估计$Q_1(\arg \max Q_2(a))$​​​，这就是**双学习（double learning）**的思想。Note that although we learn two estimates, only one estimate is updated on each play; double learning doubles the memory requirements, but does not increase the amount of computation per step. （这里自然理解的话会有些疑惑，既然要算2个估计，为什么不需要额外的计算量，只需要双倍的内存，不过通过下面的算法可以解释。）

The idea of double learning extends naturally to algorithms for full MDPs. For example, the double learning algorithm analogous to Q-learning, called **Double Q-learning**, divides the time steps in two, perhaps by flipping a coin on each step. If the coin comes up heads, the update is

![90](https://pic.imgdb.cn/item/60fbbb3f5132923bf8886e08.png)

如果硬币反面朝上，那么就交换$Q_1$和$Q_2$的角色进行同样的更新，这样就能更新$Q_2$了。<font color=red>可以看到，每一次不是$Q_1$和$Q_2$都更新，而是随机从中挑一个更新，所以计算量是不变的</font>。The two approximate value functions are treated completely symmetrically. The behavior policy can use both action-value estimates. For example, an $\epsilon$-greedy policy for Double Q-learning could be based on the average (or sum) of the two
action-value estimates.

![91](https://pic.imgdb.cn/item/60fbbb6a5132923bf88924c6.png)

## 后位状态（Afterstates）

在学习玩井字棋的时序差分方法（这个在书的第一章有提到一下），我们会发现，从中学到的函数既不是动作价值函数，也不是通常使用的状态价值函数。<font color=red>传统的价值函数估计的是在某个状态中，当智能体可以执行一个动作时，这个状态的价值，但是在井字棋中使用的状态价值函数评估的是在智能体下完一步棋之后的局面</font>。我们称之为**后位状态**，并将它们的价值函数称为**后位价值函数**。

Afterstates are useful when we have knowledge of an initial part of the environment’s dynamics but not necessarily of the full dynamics. For example, in games we typically know the immediate effects of our moves. We know for each possible chess move what the resulting position will be, but not how our opponent will reply. Afterstate value functions are a natural way to take advantage of this kind of knowledge and thereby produce a more efficient learning method.

通过井字棋的例子，我们可以清晰地看到使用后位状态来设计算法更为高效的原因。传统的动作价值将当前局面和出棋动作映射到价值的估计值，但是，很多“局面-下法”二元组都会产生相同的局面。在这样的情形下，“局面-下法”二元组是不同的，但是会产生相同的“后位状态”，因此它们的价值也必须是相同的。传统的价值函数会分别评估这两个“局面-下法”二元组，但是后位状态价值函数会将这两种情况看成一样的。任何关于下图左边的“局面-下法”二元组的学习，都会立即迁移到右边去。

![92](https://pic.imgdb.cn/item/60fbbb6a5132923bf88924dc.png)

Afterstate methods are still aptly described in terms of generalized policy iteration, with a policy and (afterstate) value function interacting in essentially the same way.

## Summary

Finally, in this chapter we have discussed TD methods entirely within the context of einforcement learning problems, but TD methods are actually more general than this. **They are general methods for learning to make long-term predictions about dynamical systems**. It was only when TD methods were analyzed as pure prediction methods, independent of their use in reinforcement learning, that their theoretical properties first came to be well understood.

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
