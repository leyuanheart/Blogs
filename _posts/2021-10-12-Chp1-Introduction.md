---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter1
subtitle:   Introduction
date:       2021-10-12
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

当我们思考何为学习的本质时，也许首先会想到：人类是通过与环境交互来学习的。婴儿玩耍时会挥舞双臂、四处张望，没有老师的指引，只用运动感知使其和外部环境直接联结。这种联结会产生大量信息，正是这些信息告诉了我们事情之间的因果关系、特定动作的后果，以及达到目标的方式。


学习者不会被告知应该采取什么动作，而是必须自己通过尝试去发现哪些动作会产生最丰厚的收益。在最困难的案例里，动作往往影响的不仅仅是即时收益，也会影响下一个情境，从而影响随后的收益。

**trial-and-error** and **delayed reward** are the two most important distinguishing features of reinforcement learning.

在这个不确定的环境中，智能体想要实现一个目标。智能体的动作会影响未来环境的状态，进而影响未来的决策和机会。因此正确的选择需要考虑到间接的、延迟的动作后果，需要有远见和规划。

In some cases the policy would be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process. The policy is the **core** of reinforcement learning agent in the sense that it alone is sufficient to determine behavior. In general, policies may be stochastic, specifying probabilities for each action.

收益(reward)信号表明了在短时间内什么是好的，而价值函数则表示了从长远的角度看什么是好的。例如，某个状态的即时收益可能很低，但它仍然可能具有很好的价值，因为之后定期会出现一些高收益的状态，反之亦然。用人来打比方，收益就像即时的愉悦（高收益）和痛苦（低收益），而价值则是在当前的环境与特定状态下，对未来我们究竟有多愉悦或多痛苦的判断。

从某种意义上说，收益更加重要，而作为收益预测的价值次之。没有收益就没有价值，而评估价值的唯一目的就是获得更多的收益。然而，在指定和评估策略时，我们最关心的是价值。动作选择是基于对价值的判断做出的。我们寻求能带来最高价值而不是最高收益的状态的动作，因为这些动作从长远来看会为我们带来最大的累计收益。不幸的是，确定价值要比确定收益难得多。收益基本上是由环境直接给予的，但是价值必须综合评估，并根据智能体在整个过程中观察到的收益序列重新估计。事实上，价值评估方法才是几乎所有强化学习算法中最重要的组成成分。

**We take the position that value functions are important for efficient search in the space of policies**. The use of value functions distinguishes reinforcement learning methods from evolutionary methods that search directly in policy space guided by evaluations of entire polices.

强化学习与其他机器学习方法最大的不同，就在于前者的训练信号是用来评估给定动作的好坏的，而不是通过给正确的动作范例来进行直接的指导。这使得主动的反复试验以试探出好的动作变得很有必要。单纯的“评估性反馈”只能表明当前采取的动作的好坏程度，但却无法确定当前采取的动作是不是所有可能性中最好或者最差的。评估性反馈依赖于当前采取的动作，即采取不同的动作会得到不同的反馈；而指导性反馈则不依赖于当前采取的动作，即采取不同的动作也会得到相同的反馈。

