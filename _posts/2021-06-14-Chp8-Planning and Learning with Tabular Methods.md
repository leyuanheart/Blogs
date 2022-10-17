---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter8
subtitle:   Planning and Learning with Tabular Methods
date:       2021-06-14
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

基于模型的方法将**规划**作为其主要组成部分，而无模型的方法则主要依赖于学习。

By a model of the environment we mean anything that an agent can use to predict how the environment will respond to its actions.

一些模型生成对所有可能的结果的描述及其对应的概率分布，这种模型被称为**分布模型**。另一种模型从所有可能性中生成一个确定的结果，这个结果通过概率分布采样得到，我们称这种模型为**样本模型**。

分布模型比样本模型更强大，因为它们可以用来生成样本模型。然而在许多场景下，获得样本模型会比获得分布模型容易的多。It would be easy to write a computer program to simulate the dice rolls and return the sum, but harder and more error-prone to figure out all the possible sums and their probabilities.

In artificial intelligence, there are two distinct approaches to planning according to our definition.第一种是**状态空间规划**，可以将它们视作一种“搜索”方法，即在状态空间中寻找最优策略或实现目标的最优决策路径。每个动作都会引发状态之间的转移，价值函数的定义都是基于状态的。另一种是**方案空间规划**，我们需要定义一些操作将一个“方案”（即一个具体的决策序列）转化成另一个，并且如果价值函数存在，它是被定义在不同的“方案”所构成的空间中的。Plan-space planning includes evolutionary methods and “partial-order planning,” a common kind of planning in artificial intelligence in which the ordering of steps is not completely determined at all stages of planning. Plan-space methods are difficult to apply efficiently to the stochastic sequential decision problems that are the focus in reinforcement learning。

本文认为，所有的状态空间规划算法都有一个通用的结构。

![126](https://pic.imgdb.cn/item/60fbc1ad5132923bf8a4dc9a.png)

不同的方法只在它们所做的回溯操作、它们执行操作的顺序，以及回溯的信息被保留的时间长短上有所不同。

学习方法和规划方法的核心都是通过回溯操作来评估价值函数。不同之处在于，规划利用模拟仿真实验产生的经验，学习方法使用环境产生的真实经验。当然，这种差异导致了许多其他的差异，例如，如何评估性能、以及如何使用环境来评估价值函数。但是通用结构意味着许多思想和算法可以在规划和学习之间迁移。特别的，在许多情况下，学习算法可以代替规划方法的关键回溯步骤。学习方法只需要输入经验，可以是真实经验，也可以是由模拟仿真的经验。

## Dyna：集成在一起的规划、动作和学习

When planning is done online, while interacting with the environment, a number of interesting issues arise. New information gained from the interaction may change the model and thereby interact with planning.

对一个规划智能体来说，实际经验至少扮演了两个角色：它可以用来改进模型（使模型与现实环境更精确地匹配）；另外，它可以直接改善价值函数和策略。前者称为**模型学习**，后者则称为**直接强化学习**（direct RL）。

![127](https://pic.imgdb.cn/item/60fbc1ad5132923bf8a4dca3.png)

直接和间接的方法各有优缺点。间接方法往往能更充分利用有限的经验，减少与环境的相互作用。直接方法则简单的多，它不受模型的设计偏差的影响。Related debates in psychology and artificial intelligence concern the relative importance of **cognition** as
opposed to **trial-and-error learning**, and of **deliberative planning** as opposed to **reactive decision making**.

![128](https://pic.imgdb.cn/item/60fbc1ad5132923bf8a4dcb5.png)

我们使用术语搜索控制（search control）来指代为模型生成的模拟经验选择初始状态和动作的过程。通常情况下，在Dyna-Q中，同样的强化学习方法既可以从真实的经验中学习，也可以应用于模拟经验已进行规划。

![129](https://pic.imgdb.cn/item/60fbc1ad5132923bf8a4dcc3.png)

### 例子：Dyna迷宫

![130](https://pic.imgdb.cn/item/60fbc1ad5132923bf8a4dce0.png)

如果没有规划（n=0），则每一幕只会给策略学习增加一次学习机会，即智能体仅仅在一个特定时刻（该幕的最后一个时刻）进行了学习。如果是n-step的方法，有效的更新也只发生在最后的n步。而规划的时候，虽然在第一幕中仍然只有一步的学习，但是到了第二幕时，可以学出一个更宽泛的策略，于是不用等到这一幕结束就可以做多次（有效的）回溯更新计算，这些计算几乎可以回溯到初始状态（这和n-step类似，只不过它不是按照固定的顺序，而是随机更新的，这样会更有效率一些）。

## 当模型错误的时候

因为在随机环境中只有数量有限的样本会被观察到，或者因为模型是通过泛化能力较差的函数来近似的，又或者仅仅是因为环境发生改变且新的动态特性尚未被观察到，模型都有可能不正确。当模型不正确时，规划过程就可能会计算出次优的策略。

在某些情况下，规划计算出来的次优策略会使得我们很快发现并修正模型错误。这种情况往往会在模型比较“乐观”的时候发生，即模型倾向于预测比真实可能情况更大的收益或更好的状态转移。规划得出的策略会尝试开发（exploit）这些机会，这样，智能体很快就能发现其实这些机会根本不存在，于是感知到模型错误，进而修正错误。

### 屏障迷宫

![131](https://pic.imgdb.cn/item/60fbc2535132923bf8a7a6b4.png)

当环境比以前更好，但以前正确的策略并没有反应出这些改善时，智能体的学习就会遇到很大困难。

### 捷径迷宫

![132](https://pic.imgdb.cn/item/60fbc2535132923bf8a7a6d3.png)

这里遇到的问题是试探和开发之间矛盾的另一种版本。在“规划”的语境中，“试探”意味着尝试那些改善模型的动作，而“开发”则意味着以当前模型的最优方式来执行动作。我们希望智能体通过试探发现环境的变化，但又不希望太多使得平均性能大大下降。

解决了捷径迷宫问题的Dyna-Q+就采用了一种启发式方法。该智能体对每一个“状态-动作”二元组进行跟踪，记录它自上一次在于环境进行真实交互时出现以来，已经过去了多少时刻。时间越长，我们就越有理由推测这个二元组相关的环境动态特性会产生变化，也即关于它的模型是不正确的。为了鼓励长期未出现过的动作，一个特殊的为该动作相关的模拟经验设置的“额外收益”将会提供给智能体。特别是，如果模型对单步转移的收益是$r$，而这个转移在$\tau$时刻内没有被尝试，那么在更新时就会采用$r+\kappa \sqrt{\tau}$的收益，其中$\kappa$是一个比较小的数字。这会鼓励智能体不断试探所有可访问的状态转移，甚至使用一长串的动作来完成试探（<font color=red>The Dyna-Q+ agent was changed in two other ways as well. First, actions that had never been tried before from a state were allowed to be considered in the planning step (f) of the Tabular Dyna-Q algorithm in the box above. Second, the initial model for such actions was that they would lead back to the same state with a reward of zero（如果不太理解可以看代码）</font>）。

##  优先遍历（Prioritized Sweeping）

前面介绍的Dyna智能体，模拟转移是从所有先前经历过的每一个“状态-动作”二元组中随机均匀采样得到的。但均匀采样通常不是最好的。For example, consider what happens during the second episode of the first maze task. At the beginning of the second episode, **only the state–action pair leading directly into the goal has a positive value**; the values of all other pairs are still zero. This means that it is pointless to perform updates along almost all transitions, because they take the agent from one zero-valued state to another, and thus the updates would have no effect. Only an update along a transition into the state just prior to the goal, or from it, will change any values. **If simulated transitions are generated uniformly, then many wasteful updates will be made before stumbling onto one of these useful ones.**

一般来说，我们希望不仅从目标状态反向计算，而且还要从能使价值函数发生变化的任何状态进行反向计算。我们假设智能体在环境中发生了一个变化，改变了它的某个状态的估计价值，通常这将意味着也应该改变许多其他状态的价值，但是唯一有用的单步更新是与特定的一些动作相关的，这些动作直接使得智能体进入价值被改变的状态。如果这些动作的价值被更新，则它们的前导状态的价值也会依次改变。若以上分析成立，那么我们就首先应该更新相关的前导动作的价值，然后改变它们的前导状态的价值。通过这种方式，我们可以从任意的价值发生变化的状态中进行反向操作，要么进行有用的更新，要么就终止信息的传播。这个总体思路可以称为**规划计算的反向聚焦**（backward focusing of planning computations）。

根据某种更新迫切性的度量对更新进行优先级排序是很自然的。这就是**优先级遍历**背后的基本思想。如果某个”状态-动作“二元组在更新之后的价值变化是不可忽略的，就将它放入优先队列维护，这个队列是按照价值改变的大小来进行优先级排序的。当队列头部的”状态-动作“二元组被更新时，也会计算它的前导”状态-动作“二元组的影响。如果这些影响大于某个较小的阈值，我们就把相应的前导”状态-动作“二元组也插入优先队列中（如果已经在队列中，则保留优先级高的那一个）。

![133](https://pic.imgdb.cn/item/60fbc2535132923bf8a7a6e9.png)

Extensions of prioritized sweeping to stochastic environments are straightforward. The model is maintained by keeping counts of the number of times each state–action pair has been experienced and of what the next states were. It is natural then to update each pair not with a sample update, as we have been using so far, but with an expected update, taking into account all possible next states and their probabilities of occurring.

优先遍历只是分配计算量以提高规划效率的一种方式，这可能不是最好的方法。优先遍历的局限之一是它采用期望更新，这在随机环境中可能会浪费大量计算来处理低概率的转移。在许多情况下，采样更新方法可以用更小的计算量得到更接近真实价值函数的估计，虽然它会引入一些估计方差。

另一种思路就是从当前策略下经常能访问到的状态出发，估计哪些后继状态是容易到达的。然后聚焦于那些容易到达的状态。这可以称为**前向聚焦**（forward focusing）。

## 期望更新与采样更新的对比

我们已经介绍了很多种不同类型的价值函数更新。这一节我们关注单步更新方法，它们主要沿着三个维度产生变化。一个是它们更新状态价值还是动作价值；第二是它们所估计的价值对应的是最优策略还是任意给定的策略。这连个维度组合出四类更新，用于近似四个价值函数：$q^\*,v^\*,q_{\pi},v_{\pi}$​。最后一个维度是采样期望更新还是采样更新。期望更新会考虑所有可能发生的转移，而采样更新则仅仅考虑采样得到的单个转移样本。这三个维度产生了8中情况。其中7中对应于特定的算法。这些单步更新中的任何一种都可以用于规划。

![134](https://pic.imgdb.cn/item/60fbc2535132923bf8a7a702.png)

期望更新肯定会产生更好的估计，因为它们没有收到采样误差的影响，但是它们也需要更多的计算。要正确评估期望更新和采样更新的相对优势，我们必须控制不同的计算量需求。

我们以$q^\*$​​为例。假设状态和动作都是离散的，近似价值函数Q采用表格来表示，环境模型的估计采用$p(s',r\|s,a)$​​来表示。

期望更新：

![135](https://pic.imgdb.cn/item/60fbc2535132923bf8a7a716.png)

采样更新，给定$S',R$（来自模型）

![136](https://pic.imgdb.cn/item/60fbc2805132923bf8a86937.png)

当环境是随机环境时，期望更新和采样更新之间的差异是显著的。期望更新的一个优势是它精确的计算，这使得一个新的$Q(s,a)$​的正确性仅仅受限于$Q(s',a')$​在后继状态下的正确性。而采样更新还会受到采样误差的影响。但另一方面，采样更新的计算复杂度更低。对于一个特定的初始二元组$s,a$​，设$b$​是分支因子（branching factor）（即对于所有可能的后继状态$s'$​，其中$\hat{p}(s'\|s,a)>0$​的数目），则这个二元组的期望更新需要的计算量大约是采样更新的$b$​倍。

If there is enough time to complete an expected update, then the resulting estimate is generally better than that of b sample updates because of the absence of sampling error. 但是如果没有足够的时间来完成一次期望更新，那么采样更新总是比较可取的，因为它们至少可以通过少于$b$次的更新来改进估计值。With so many state–action pairs, expected updates of all of them would take a very long time. Before that we may be much better o↵ with a few sample updates at many state–action pairs than with expected updates at a few pairs. **那么给定一个单位的计算量，是否采用期望更新比采用$b$次采样更新会更好呢？**

下图显示了在多个不同的分支因子b下，期望更新和采样更新的估计误差对计算次数的函数变化。这里讨论的情况是：所有b个后继状态都是等概率发生的，并且其中初始估计的误差是1. <font color=red>后继状态的价值假定是正确的，所有期望更新在完成时会将误差减小到0. 在这种情况下，采样更新会使得误差以比例系数$\sqrt{\frac{b-1}{b\times t}}$减小，其中t是已经进行过的采样更新的次数（假定采样平均值，即$\alpha=1/t$）</font>。下图显示出一个关键现象是，对于中等大小的b来说，估计误差的大幅下降只需要消耗很少的一部分计算次数。在这些情况中，许多“状态-动作”二元组都可以使其价值得到显著的改善，甚至可以达到和期望更新差不多的效果；但在同样的时间里，却只有一个“状态-动作”二元组可以得到期望更新。（For these cases, many state–action pairs could have their values improved dramatically, to within a few percent of the effect of an expected update, in the same time that a single state–action pair could undergo an expected update.）

![137](https://pic.imgdb.cn/item/60fbc2805132923bf8a8696c.png)

上图中的采样更新的优势很可能被低估。在一个真实的问题中，后继状态的价值自身就是不断被更新的估计值。由于后继状态的价值经过更新后变得更准确，因此回溯之后的状态价值也会更快地变得准确，这是采样更新具有的第二个优势。这些都表明，对拥有很大的随机分支因子并要求对大量状态价值准确求解的问题，采样更新可能比期望更新要好。

## 轨迹采样

在许多任务中，绝大多数状态是无关紧要的，因为只有在非常糟糕的策略中才能访问到这些状态，或者访问到这些状态的概率很低。穷举式遍历将计算时间均匀地用于状态空间的所有部分，而不是集中在需要的地方。基于动态规划的经典方法是遍历整个状态（或“状态-动作”二元组）空间。虽然穷举式遍历以及对所有状态均匀分配时间的处理并非动态规划的必要属性（原则上，价值函数的更新所使用的算力可以用任何方式分配（为确保收敛，所有“状态-动作”二元组必须被无限次访问，但也有例外）），但在实践中，通常使用穷举式遍历。

另一种方法是根据某些分布从状态或“状态-动作”二元组空间采样。比如Dyna-Q中的随机采样，但是这就会遇上和穷举式遍历相同的问题。更让人感兴趣的是根据**同轨策略下的分布**来分配算力，也就是根据当前策略下观察到的状态或“状态-动作”二元组的分布来分配算力。One advantage of this distribution is that it is easily generated; one simply interacts with the model, following the current policy. In an episodic task, one starts in a start state (or according to the starting-state distribution) and simulates until the terminal state. In a continuing task, one starts anywhere and just keeps simulating. In either case, sample state transitions
and rewards are given by the model, and sample actions are given by the current policy. We call this way of generating experience and updates trajectory sampling.

如果要根据同轨策略分布（即遵循同轨策略所产生的状态和动作序列的概率分布）来进行算力的分配，除了对完整的轨迹进行采样外，很难想象出其他有效的方式。因为找到同轨策略分布的显示表达式几乎是不可能的。每当策略发生变化，分布也会发生变化，而且计算分布本身需要的算力几乎会与完整的策略评估相当。

Is the on-policy distribution of updates a good one? Intuitively it seems like a good choice, at least better than the uniform distribution. **而且当使用函数逼近时，同轨策略分布具有显著的优势。**Whether or not function approximation is used, one might expect on-policy focusing to significantly improve the speed of planning.

聚焦于同轨策略分布可能是有益的，因为这使得空间中大量不重要的区域被忽略。但这也可能是由有害的，因为这使得空间中的某些相同的旧区域一次又一次地被更新。

![138](https://pic.imgdb.cn/item/60fbc2805132923bf8a86995.png)



图上半部分显示了超过200个采样任务的平均结果，每个任务有1000个状态，分支因子分别为1、3、10.图中画的是策略质量对期望更新次数的函数。在所有情况下，根据同轨策略分布进行抽样都显示出类似的趋势：<font color=red>初始阶段规划速度更快，但长期看却变化迟缓</font>。分支因子越小时，这个效应就越强。比如图下半部分显示了分支为1的10000个状态的任务的结果。在这种情况下，同轨策略的优势很大且可以保持很长时间。

All of these results make sense. 在短期内，根据同轨策略分布进行采样有助于聚焦于接近初始状态的后继状态。如果有很多状态且分支因子很小，则这种效应会很大而且持久。从长期来看，聚焦于同轨策略分布则可能存在危害，因为通常发生的各种状态都已经有了正确的估计值。采样到它们没有什么用，只有采样到其他的撞他才会使得更新过程有价值。这可能就是从长远看穷举式的、分散的采样方式表现得更好的原因，至少对小问题来说是如此。

## 实时动态规划

Real-time dynamic programming, or RTDP, is an on-policy trajectory-sampling version of the value-iteration algorithm of dynamic programming (DP). RTDP是异步DP算法的一个例子。Asynchronous DP algorithms are not organized in terms of systematic sweeps of the state set; they update state values in any order whatsoever, using whatever values of other states happen to be available. In RTDP, the update order is dictated by the order states are visited in real or simulated trajectories.

如果轨迹只能从指定的一组初始状态开始，并且如果你对给定策略下的预测问题感兴趣，则同轨策略轨迹采样允许算法完全跳过那些给定策略从任何初始状态开始都无法到达的状态。对于一个控制问题，其目标是找到最优策略。**这时可能存在一些状态，来自任何初始状态的任何最优策略都无法到达它们**。对于这些不相关的状态，我们就不需要指定最优动作。我们需要的是**最优部分策略**，即策略对于相关状态是最优的，而对于不相关状态则可以指定任意的甚至是未定义的动作。

![139](https://pic.imgdb.cn/item/60fbc2805132923bf8a869aa.png)

但是，采取同轨策略下的轨迹采样控制方法（如Sarsa）通常需要无数次访问所有的“状态-动作”二元组——也包括哪些不相关的状态。这可以通过使用“试探性出发”来完成。

RTDP最有趣的是，对于满足某些合理条件的特定类型问题，RTDP可以保证找到相关状态下的最优策略，而无需频繁访问每个状态，甚至可以完全不访问某些状态。

RTDP相比于传统价值迭代的另一个优点是，在实践中，当价值函数在某次遍历中的该变量很小时，价值迭代终止。这时，价值函数已经非常接近$v^\*$，贪心策略也非常接近最优策略。然而，其实远在价值迭代终止之前，那些对最新的价值函数贪心的策略就有可能已经是最优的了（最优策略对于多个不同价值函数而言都可以是贪心的，而不仅仅是$v^*$）。在价值迭代收敛之前检查最优策略的出现不是传统DP算法的一部分，并且需要大量额外的计算。

## 决策时规划

规划过程至少有两种运行方式。之前介绍的是以动态规划和Dyna为代表的方法。它们从环境模型（单个样本或概率分布）生成模拟经验，并以此为基础采用规划逐步改进策略或价值函数。在为当前状态$S_t$进行动作选择前，规划过程都会预先针对多个状态（包括$S_t$）的动作选择所需要的表格条目（表格型方法）或数学表达式（近似方法）进行改善。在这种运行方式下，规划并不仅仅聚焦于当前状态，而是还要预先在后台处理其他多个状态。这种叫**后台规划（background planning）**。

另一种规划的运行方式是在遇到每个新状态$S_t$之后才开始并完成规划，<font color=red>这个过程的输出是单个动作$A_t$</font>，在下一时刻，规划过程将从一个新的状态$S_{t+1}$开始，产生$A_{t+1}$，以此类推。这里的规划聚焦于特定的状态，我们称之为决策时规划（decision-time planning）。

<font color=blue>我们可以用两种方式来解读“规划”——一是使用模拟经验来逐步改进策略或价值函数；二是使用模拟经验来为当前状态选择一个动作。</font>

Even when planning is only done at decision time, we can still view it as proceeding from simulated experience to updates and values, and ultimately to a policy. 只不过现在的情况是，价值和策略不是普适的，而是对于当前状态和可选动作是特定的。**因此，在选择了当前状态下的动作后，我们通常可以丢弃规划过程中创建的价值和策略**。在许多应用中，这并不是一个巨大的损失，因为一个应用往往存在很多状态，而我们不太可能在短时间内回到同一个状态。In general, one may want to do a mix of both: focus planning on the current state and store the results of planning so as to be that much farther along should one return to the same state
later. 决策时规划在不需要快速响应的应用程序中是最有用的，比如下棋。

## 启发式搜索（Heuristic Search）

人工智能中经典的状态空间规划方法就是决策时规划，整体被统称为**启发式搜索**。在启发式搜索中，对于遇到的每个状态，我们建立一个树结构，该结构包含了后面各种可能的延续。我们将近似价值函数应用于叶子节点（如果叶子节点就是终止状态的话，则不需要价值函数了），然后向根部的当前状态回溯更新。计算了这些节点的更新值之后，选择其中最好的的值作为当前动作，<font color=red>然后舍弃所有更新值</font>。

The point of searching deeper than one step is to obtain better action selections. 如果有一个完美的环境模型和一个不完美的动作价值函数，那么实际上更深的搜索通常会产生更好的策略。如果搜索一直持续到整幕序列结束的位置，那么不完美的价值函数的影响就被消除了。或者搜索足够的深度$k$使得$\gamma^k$非常小，则价值函数的影响也可以忽略不计。

我们应该能够看到启发式搜索中算力聚焦的最明显的方式：关注当前状态。

![140](https://pic.imgdb.cn/item/60fbc2805132923bf8a869c5.png)

## 预演（Rollout）算法

预演算法是一种基于蒙特卡洛控制的决策时规划算法，这里的蒙特卡洛控制应用于以当前环境状态为起点的采样模拟轨迹。

the goal of a rollout algorithm is not to estimate a complete optimal action-value function, $q^\*$​, or a complete action-value function, $q_{\pi}$​, for a given policy $\pi$​. 相反，它们仅对每一个当前状态以及一个给定的被称为**预演策略**的策略做出动作价值的蒙特卡洛估计。作为一种决策时规划算法，预演算法只是即时地利用这些动作价值的估计值，之后就丢弃它们。

预演算法的目的是为了改进预演策略的性能，而不是找到最优的策略。从过去的经验来看，预演算法表现得出人意料地有效。在一些应用场合下，即使预演策略是完全随机的，预演算法也能产生很好的效果。But the performance of the improved policy depends on properties of the rollout policy and the ranking of actions produced by the Monte Carlo value estimates. Intuition suggests that the better the rollout policy and the more accurate the value estimates, the better the policy produced by a rollout algorithm is
likely be.[Combining Online and Offline Knowledge in UCT, Gelly and Silver, 2007](https://hal.inria.fr/inria-00164003/document)

这就涉及了一个重要的权衡，因为更好的预演策略一般都意味着要花更多的时间来模拟足够多的轨迹从而得到一个良好的价值估计。

Because the Monte Carlo trials are independent of one another, it is possible to run many trials in parallel on separate processors. Another approach is to truncate the simulated trajectories short of complete episodes, correcting the truncated returns by means of a stored evaluation function (which brings into play all that we have said about truncated returns and updates in the preceding chapters).

我们一般不认为预演算法是一种**学习**算法，因为他们不保持对于价值或策略的长时记忆。但是这类算法利用了强化学习的一些特点。比如：通过轨迹采样来避免对于动态规划的穷举式遍历，已经使用采样更新而非期望更新来避免使用概率分布模型。

## 蒙特卡洛树搜索

Monte Carlo Tree Search (MCTS)是一种预演算法，**但是它的进步之处在于引入了一种新手段，通过积累蒙特卡洛模拟得到的价值估计来不停地将模拟导向高收益轨迹**。

MCTS的核心思想对从当前状态出发的多个模拟轨迹不断地聚焦和选择，这是通过**扩展**模拟轨迹中过的较高评估值的初始片段来实现的，而这些评估值则是根据更早的模拟样本计算的。

MCTS does not have to retain approximate value functions or policies from one action selection to the next, though in many implementations it retains selected action values likely to be useful for its next execution.

在计算过程中，我们只维护部分的蒙特卡洛估计值，这些估计值对应于会在几步之内到达的“状态-动作”二元组所形成的子集，这就形成了一棵以当前状态为根节点的树，如下图所示。MCTS会增量式地逐渐添加节点来进行树扩展，这些节点代表了从模拟轨迹的结果上看前景更为光明的状态。任何一条模拟轨迹都会沿着这棵树运行，最后从某个叶子节点离开。在树的外部以及叶子节点上，通过预演策略选择动作，对树内部的状态可能有更好的方法。对于内部状态，至少对部分动作有价值估计，所以我们可以用一个知晓这些信息的策略来从中选择。这个策略可以被称为树策略，它能够平衡试探和开发。比如$\epsilon$-贪心策略或者UCB选择规则。

![141](https://pic.imgdb.cn/item/60fbc2b15132923bf8a939b5.png)

![142](https://pic.imgdb.cn/item/60fbc2b15132923bf8a939d6.png)

MCTS continues executing these four steps, starting each time at the tree’s root node, until no more time is left, or some other computational resource is exhausted. Then, finally, an action from the root node (which still represents the current state of the environment) is selected according to some mechanism that depends on the accumulated statistics in the tree; for example, it may be an action having the largest action value of all the actions available from the root state, or perhaps the action with the largest visit count to avoid selecting outliers. This is the action MCTS actually selects.

在环境转移到一个新状态后，MCTS会再次执行。有时候会重新构造一棵树，根节点就是新的状态，但是通常情况下会使用上次执行MCTS时构造的树的一部分，仅保留新状态以及它的一级子节点的信息，其他节点以及相应的动作值都舍弃掉。

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
