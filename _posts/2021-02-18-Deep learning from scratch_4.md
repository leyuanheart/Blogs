---
layout:     post
title:      Deep learning from scratch 4
subtitle:   Learning Tricks
date:       2021-02-18
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Deep Learning
---

<img src="https://img.imgdb.cn/item/60311e2d5f4313ce25b859d6.jpg" alt="fengmian" style="zoom:50%;" />

私认为这是这本书让我收获最大的部分。

# 与学习相关的技巧

本篇将介绍神经网络的学习中的一些重要观点，主题涉及寻找最优权重参数的最优化方法、权重参数的初始值、超参数的设定方法等。此外，为了应对过拟合，本章还将介绍权值衰减、Dropout等正则化方法，并进行实现。最后将对近年来众多研究中使用的 Batch Normalization方法进行简单的介绍。

## 参数的更新

神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization)。遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。 

之前我们使用的是随机梯度下降法（stochastic gradient descent），称为SGD。SGD是一个简单的方法，不过比起胡乱地搜索参数空间，也算是“聪明”的方法。但是，根据不同的问题，也存在比SGD更加聪明的方法。

下面这个对于SGD的比喻我觉得非常形象，所以我引用一下。

> 有一个性情古怪的探险家。他在广裹的干旱地带旅行，坚持寻找幽深的山谷。他的目标是要到达最深的谷底（他称之为“至深之地”）。这也是他旅行的目的。并且， 他给自己制定了两个严格的“规定”：一个是不看地图；另一个是把眼睛蒙上。因此， 他并不知道最深的谷底在这个广裹的大地的何处，而且什么也看不见。在这么严苛的条件下，这位探险家如何前往“至深之地”呢？他要如何迈步，才能迅速找到“至深之地”呢？ 
> 寻找最优参数时，我们所处的状况和这位探险家一样，是一个漆黑的世界。我们必须在没有地图、不能睁眼的情况下，在广裹、复杂的地形中寻找“至深之地”。大家可以想象这是一个多么难的问题。 
> 在这么困难的状况下，地面的坡度显得尤为重要。探险家虽然看不到周围的情况，但是能够知道当前所在位置的坡度（通过脚底感受地面的倾斜状况）。于是，朝着当前所在位置的坡度最大的方向前进，就是SGD的策略。勇敢的探险家心里可能想着只要重复这一策略，总有一天可以到达“至深之地”。 

### SGD

回顾一下SGD，用数学式表达：


$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$


我们将SGD实现为一个Python类（为方面后面使用）

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

这里，进行初始化时的参数lr表示learning rate（学习率）。这个学习率会保存为实例变量。此外，代码段中还定义了 update(params, grads）方法，这个方法在SGD中会被反复调用。参数params和grads（与之前的神经网络的实现一样）是字典型变量， 按params[' W1' ]、grads[ 'W1'] 的形式，分别保存了权重参数和它们的梯度。 

使用这个SGD类，可以按照如下方式进行神经网络的参数更新（省略了一些中间步骤）

```python
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    grads = network.gradient(x, t)
    params = network.params
    params = optimizer.update(params, grads)   # params改变，network.params也会跟着改变
```

这里首次出现的变量名optimizer。表示“进行最优化的人”的意思，这里由SGD承担这个角色。参数的更新由optimizer负责完成。我们在这里需要做的只是将参数和梯度 的信息传给optimizer。

像这样，通过单独实现进行最优化的类，功能的模块化变得更简单。比如．后面我们马上会实现另一个最优化方法Momentum，它同样会实现成拥有。update( params, grads）这个共同方法的形式。这样一来，只需要将optimizer=SGD(）这一语句换成 optimizer=Momentum()，就可以从SGD切换为Momentum。 

### SGD的缺点

虽然SGD简单，并且容易实现，但是在解决某些问题时可能没有效率。我们考虑求如下函数的最小值的问题。


$$
f(x, y) = \frac{1}{20}x^2+y^2
$$


![48](https://img.imgdb.cn/item/6031c6745f4313ce25eb856e.png)

如上图所示，函数是向x轴方向延伸的“碗”状函数。等高线呈向x轴方向延伸的椭圆状。

接下来看一下它的梯度图。从下图可以看出，梯度在y轴方向上大，x轴方向上小。这里需要注意的是，虽然函数的最小值在$(x,y)=(0,0)$处，但是图中很多梯度的方向并没有指向$(0,0)$。

![49](https://img.imgdb.cn/item/6031c6745f4313ce25eb8570.png)

我们来尝试该函数应用SGD。从$(x,y) = (-7.0, 2.0)$处（初始值）开始搜索，结果如下图所示。 
在图中，SGD呈“之”字形移动。这是一个相当低效的路径。也就是说，SGD的缺点是，如果函数的形状非均向(anisotropic)，比如呈延伸状，搜索的路径就会非常低效。因此，我们需要比单纯朝梯度方向前进的SGD更聪明的方法。**SGD低效的根本原因是，梯度的方向并没有指向最小值的方向**。 为了改正SGD的缺点，下面我们将介绍Momentum、 AdaGrad和Adam这3种方法来取代SGD。

![50](https://img.imgdb.cn/item/6031c6745f4313ce25eb8573.png)

```python
import numpy as np
import matplotlib.pyplot as plt

def func(x,y):
    return x**2/20 + y**2

def g_f(x, y):
    return np.array([x/10, 2*y])

x = np.arange(-10, 11., 0.5)
y = np.arange(-5, 6., 0.5)

X, Y = np.meshgrid(x, y)
Z = func(X, Y)


# 等高线图

plt.figure()
c = plt.contour(X, Y, Z, levels=50)

# 三维图

fig = plt.figure()
axe = plt.axes(projection='3d')
axe.contour3D(X, Y, Z, levels=100)


# 梯度图

grads = g_f(X.flatten(), Y.flatten())
plt.figure()
plt.quiver(X.flatten(), Y.flatten(), -grads[0], -grads[1],  angles="uv",color="#666666")



# 训练

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizer = SGD(lr=0.95)

x_history = []
y_history = []
    
for i in range(30):
    x_history.append(params['x'])
    y_history.append(params['y'])
    
    grads['x'], grads['y'] = g_f(params['x'], params['y'])
    optimizer.update(params, grads)
    
    
    
x = np.arange(-10, 10, 0.01)
y = np.arange(-5, 5, 0.01)

X, Y = np.meshgrid(x, y) 
Z = func(X, Y)

# for simple contour line 

mask = Z > 7
Z[mask] = 0

plt.plot(x_history, y_history, 'o-', color="red")
plt.contour(X, Y, Z)
plt.ylim(-10, 10)
plt.xlim(-10, 10)
plt.plot(0, 0, '+')
plt.title('SGD')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

### Momentum

Momentum是“动量”的意思，和物理有关。用数学式表示Momentum方法为：


$$
\begin{align}
v \leftarrow \alpha v + \eta \frac{\partial L}{\partial W} \\
W \leftarrow W - v
\end{align}
$$


和前面的SGD一样，W表示要更新的权重参数，$\frac{\partial L}{\partial W}$表示损失函数关于W的梯度， $\eta$表示学习率。这里新出现了一个变量$v$。对应物理上的速度。第一个式子表示了物体在梯度方向上受力，在这个力的作用下，物体的速度增加这一物理法则。如下图所示， Momentum方法给人的感觉就像是小球在地面上滚动。 

![51](https://img.imgdb.cn/item/6031caf55f4313ce25ed1e50.png)

$\alpha v$这一项。在物体不受任何力时，该项承担使物体逐渐减速的任务 (a设定为0.9之类的值），对应物理上的地面摩擦或空气阻力。

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * grads[key]
            params[key] -= self.v[key]
```

现在尝试使用Momentum解决之前的最优化问题，只需要把optimizer换成Momentum就行了，结果如下图。 

![52](https://img.imgdb.cn/item/6031caf55f4313ce25ed1e55.png)

图中，更新路径就像小球在碗中滚动一样。和SGD相比，我们发现“之”字形 的“程度”减轻了。这是因为虽然x轴方向上受到的力非常小，但是一直在同一方向上受力，所以朝同一个方向会有一定的加速。反过来，虽然y轴方向上受到的力很大，但是因 为交互地受到正方向和反方向的力，它们会互相抵消，所以y轴方向上的速度不稳定。因此，和SGD时的情形相比，可以更快地朝x轴方向靠近，减弱“之”字形的变动程度。 

### AdaGrad

在神经网络的学习中，学习率（数学式中记为lr)的值很重要。学习率过小，会导致学习花费过多时间；反过来，学习率过大，则会导致学习发散而不能正确进行。 

在关于学习率的有效技巧中，有一种被称为**学习率衰减**（(learning rate decay）的方法，即随着学习的进行，使学习率逐渐减小。实际上，一开始“多”学，然后逐渐“少”学的方法，在神经网络的学习中经常被使用。 

逐渐减小学习率的想法，相当于将“全体”参数的学习率值一起降低。而AdaGrad进一步发展了这个想法，针对“一个一个”的参数，赋予其“定制”的值。 

AdaGrad会为参数的每个元素适当地调整学习率，与此同时进行学习（AdaGrad的 Ada来自英文单词Adaptive，即“适当的”的意思）。下面，让我们用数学式表示AdaGrad 的更新方法。 


$$
\begin{align}
h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W} \\
W \leftarrow W - \eta \frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}
\end{align}
$$



这里新出现了变量h，它保存了以前的所有梯度值的平方和（$\odot$表示对应矩阵元素的乘法(element-wise)）。然后，在更新参数时，通过乘以$\frac{1}{\sqrt{h}}$ ，就可以调整学习的尺度。这意味着，参数的元素中变动较大（被大幅更新）的元素的学习率将变小。也就是说，可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐 渐减小。 

>  AdaGrad会记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小。实际上，如果无止境地学习，更新量就会变为0，完全不再更新。为了改善这个问题，可以使用RMSProp方法。RMSProp方法并不是将过去所有的梯度一视同仁地相加，而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反映出来。这种操作从专业上讲，称为“指数移动平均”，呈指数函数式地减小过去的梯度的尺度。 

现在来实现AdaGrad。

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)
```

这里需要注意的是，最后一行加上了微小值1e-7。这是为了防止当self.h[key] 中有0 时，将0用作除数的情况。在很多深度学习的框架中，这个微小值也可以设定为参数，但这里我们用的是1e-7这个固定值。 

![53](https://img.imgdb.cn/item/6031caf55f4313ce25ed1e5b.png)

函数的取值高效地向着最小值移动。由于y轴方向上的梯度较大，因此刚开始变动较大，但是后面会根据这个较大变动按比例进行调整，减小更新的步伐。因此，y轴方向的更新程度被减弱，“之”字形的变动程度有所衰减。 

### Adam

Momentum参照小球在碗中滚动的物理规则进行移动，AdaGrad为参数的每个元素适当地调整更新步伐。如果将这两个方法融合在一起会怎么样呢？这就是Adam方法的基本思路.

 Adam是2015年提出的新方法。它的理论有些复杂，直观地讲，就是融合了 Momentum和AdaGrad的方法。通过组合前面两个方法的优点，有望实现参数空间的高效搜索。此外，进行超参数的“偏置校正”也是Adam的特征。这里不再进行过多的说明， 详细内容请参考[原作者的论文](https://arxiv.org/pdf/1412.6980.pdf)。

> 这里关于Adam方法的说明只是一个直观的说明，并不完全正确。详细内容请参考原作者的论文。

```python
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

>  Adam会设置3个超参数。一个是学习率（论文中以$\alpha$出现），另外两个是一次momentum系数$\beta_1$和二次momentum系数$\beta_2$。根据论文，标准的设定值是0.9和0.999。设置了这些值后，大多数情况下都能顺利运行。 

![54](https://img.imgdb.cn/item/6031caf55f4313ce25ed1e61.png)

在图中，基于Adam的更新过程就像小球在碗中滚动一样。虽然Momentun也有类似的移动，但是相比之下，Adam的小球左右摇晃的程度有所减轻。这得益于学习的更新程度被适当地调整了。 

### 使用哪种更新方法？

上面我们介绍了SGD、Momentum、 AdaGrad、 Adam这4种方法，那么用哪种方法好呢？非常遗憾，（目前）并不存在能在所有问题中都表现良好的方法。这4种方法各有各的特点，都有各自擅长解决的问题和不擅长解决的问题。 

很多研究中至今仍在使用SGD 。Momentum和AdaGrad也是值得一试的方法。 最近，很多研究人员和技术人员都喜欢用Adam。

## 权重的初始值

在神经网络的学习中，权重的初始值特别重要。实际上，设定什么样的权重初始值， 经常关系到神经网络的学习能否成功。本节将介绍权重初始值的推荐值，并通过实验确认神经网络的学习是否会快速进行。 

### 可以将权重初始值设为0吗？

后面我们会介绍抑制过拟合、提高泛化能力的技巧——权值衰减（weight decay)。 简单地说，权值衰减就是一种以减小权重参数的值为目的进行学习的方法。通过减小权重参数的值来抑制过拟合的发生。 

如果想减小权重的值，一开始就将初始值设为较小的值才是正途。实际上，在这之前 的权重初始值都是像0.01$*$np.random.rancin(10, 100）这样，使用由高斯分布生成 的值乘以0.01后得到的值（标准差为0.01的高斯分布）。 

如果我们把权重初始值全部设为0以减小权重的值，会怎么样呢？从结论来说，将权重初始值设为0不是一个好主意。事实上，将权重初始值设为0的话，将无法正确进行学习。

为什么不能将权重初始值设为0呢？**严格地说，为什么不能将权重初始值设成一样的值呢**？这是因为在误差反向传播法中，所有的权重值都会进行相同的更新。比如，在2层神经网络中，假设第1层和第2层的权重为0。这样一来，正向传播时，因为输入层的权重为0，所以第2层的神经元全部会被传递相同的值。第2层的神经元中全部输入相同的值，这意味着反向传播时第2层的权重全部都会进行相同的更新（回忆一下“乘法节点的反向传播”的内容）。因此，权重被更新为相同的值，并拥有了对称的值（重复的值）。 这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化”（严格地讲， 是为了瓦解权重的对称结构），必须随机生成初始值。 

举个小例子吧。

![90](https://img.imgdb.cn/item/603266f95f4313ce252eb2c3.png)

### 隐藏层的激活值分布

观察隐藏层的激活值（激活函数的输出数据）的分布，可以获得很多启发。这里， 我们来做一个简单的实验，观察权重初始值是如何影响隐藏层的激活值的分布的。这里要做的实验是，向一个5层神经网络（激活函数使用sigmoid函数）传入随机生成的输入数据，用直方图绘制各层激活值的数据分布。

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
input_data = np.random.randn(1000, 100)  # 1000个数据

node_num = 100  # 各隐藏层的节点（神经元）数

hidden_layer_size = 5  # 隐藏层有5层

activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 权重初始化
    
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    z = np.dot(x, w)

    a = sigmoid(z)


    activations[i] = a

# 可视化

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    
    # plt.ylim(0, 7000)
    
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
```

这里假设神经网络有5层，每层有100个神经元。然后，用高斯分布随机生成1000 个数据作为输入数据，并把它们传给5层神经网络。激活函数使用sigmoid函数，各层的 激活的结果保存在activations变量中。这个代码段中需要注意的是权重的尺度。虽然这次我们使用的是标准差为1的高斯分布，但实验的目的是通过改变这个尺度（标准差），观察激活值的分布如何变化。现在，我们将保存在activations中的各层数据画成直方图。 

![56](https://img.imgdb.cn/item/6031cbaf5f4313ce25ed59be.png)

从图可知，各层的激活值呈偏向0和1的分布。这里使用的sigmoid函数是S 型函数，随着输出不断地靠近0（或者靠近1)，它的导数的值逐渐接近0。因此，偏向0 和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。这个问题称为**梯度消失**（gradient vanishing)。层次加深的深度学习中，梯度消失的问题可能会更加严重。 

下面，将权重的标准差设为0.01，进行相同的实验。实验代码只需要把设定权重初始值的地方相应的替换就行了。

![57](https://img.imgdb.cn/item/6031cbaf5f4313ce25ed59c3.png)

这次呈集中在0.5附近的分布。因为不像刚才的例子那样偏向0和1，所以不会发生梯度消失的问题。但是，激活值的分布有所偏向，说明在表现力上会有很大问题。为什么这么说呢？因为如果有多个神经元都输出几乎相同的值，那它们就没有存在的意义了。比如，如果100个神经元都输出几乎相同的值，那么也可以由1个神经元来表达基本相同的事情。因此，激活值在分布上有所偏向会出现“表现力受限”的问题。 

> 各层的激活值的分布都要求有适当的广度。为什么呢？因为通过在各层间传递多样性的数据，神经网络可以进行高效的学习。反过来，如果传递的是有所偏向的数据，就会出现梯度消失或者“表现力受限”的问题，导致学习可能无法顺利进行。 

接着，我们尝试使用Xavier Glorot等人的论文中推荐的权重初始值（俗称“Xavier ”初始值）。现在，在一般的深度学习框架中，Xavier初始值己被作为标准使用。
Xavier的论文中，为了使各层的激活值呈现出具有相同广度的分布，推导了合适的权重尺度。推导出的结论是，如果前一层的节点数为n，则初始值使用标准差为$\frac{1}{\sqrt{n}}$的分布。 

> Xavier的论文中提出的设定值，不仅考虑了前一层的输入节点数量，还考虑了下一层的输出节点数量。但是， Caffe等框架的实现中进行了简化，只使用了这里所说的前一层的输入节点进行计算。  

![58](https://img.imgdb.cn/item/6031cbb05f4313ce25ed59c9.png)

使用Xavier初始值后，前一层的节点数越多，要设定为目标节点的初始值的权重尺度就越小。代码同样只需要改动初始化的地方。

![59](https://img.imgdb.cn/item/6031cbb05f4313ce25ed59d1.png)

使用Xavier初始值后的结果如图所示。从这个结果可知，越是后面的层，图像变得越歪斜，但是呈现了比之前更有广度的分布。因为各层间传递的数据有适当的广度， 所以sigmoid函数的表现力不受限制，有望进行高效的学习。 

后面的层的分布呈稍微歪斜的形状。如果用tanh函数 （双曲线函数）代替sigmoid函数，这个稍微歪斜的问题就能得到改善。实际上， 使用tanh函数后，会呈漂亮的吊钟型分布。tanh函数和sigmoid函数同是S型曲 线函数，但tanh函数是关于原点（0, 0）对称的S型曲线，而sigmoid函数是关于 (x,y)=(0, 0.5)对称的S型曲线。**众所周知，用作激活函数的函数最好具有关于原点对称的性质**（我还真不知道...）。 

### ReLU的权重初始值 

Xavier初始值是以激活函数是线性函数为前提而推导出来的。因为sigmoid函数和 tanh函数左右对称，且中央附近可以视作线性函数，所以适合使用Xavier初始值。但当 激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是Kaiming He等人推荐的初始值，也称为“He初始值”。当前一层的节点数为n时，He初始值使用标准差为$\sqrt{\frac{2}{n}}$的高斯分布。（直观上）可以解释为，因为ReLU的负值区域的值为0，为了使它更有广度， 所以需要2倍的系数。 

现在来看一下激活函数使用ReLU时激活值的分布。我们给出了3个实验的结果，依次是权重初始值为标准差是0.01的高斯分布（下文简写为“std = 0.01"）时、初始值为Xavier初始值时、初始值为ReLU专用的“He初始值”时的结果。 

![60](https://img.imgdb.cn/item/6031cbb05f4313ce25ed59da.png)

观察实验结果可知，当“std=0.01”时，各层的激活值非常小。神经网络传递的是非常小的值。说明反向传播时权重的梯度也很小，这是很严重的问题，实际上学习基本上没什么进展。

接下来是初始值为Xavier初始值时的结果。在这种情况下，随着层的加深，偏向一点点变大。实际上，层加深后，激活值的偏向变大，学习时会出现梯度消失的问题。而当初始值为He初始值时，各层中分布的广度相同。由于即便层加深，数据的广度也能保持 变，因此反向传播时，也会传递合适的值。

 总结一下，当激活函数使用ReLU时，权重初始值使用He初始值，当激活函数为 sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值。这是目前的最佳实践。 

**权重的初始化很重要！！！**

## Batch Normalization

在上一节，我们观察了各层的激活值分布，并从中了解到如果设定了合适的权重初始 值，则各层的激活值分布会有适当的广度，从而可以顺利地进行学习。那么，为了使各层拥有适当的广度， “强制性”地调整激活值的分布会怎样呢？实际上，Batch Normalization方法就是基于这个想法而产生的。 

为什么Batch Norm这么惹人注目呢？因为Batch Norm有以下优点。

- 可以使学习快速进行（可以增大学习率）。 

- 不那么依赖初始值（对于初始值不用那么神经质） 。

- 抑制过拟合（降低Dropout等的必要性）。 

考虑到深度学习要花费很多时间，第一个优点令人非常开心。另外，后两点也可以帮我们消除深度学习的学习中的很多烦恼。 

如前所述，Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度。为 此，要向神经网络中插入对数据分布进行正规化的层，即Batch Normalization层（下文简 称Batch Norm层），如图。 

![61](https://img.imgdb.cn/item/6031cbe25f4313ce25ed6cbc.png)

Batch Norm，顾名思义，以进行学习时的mini-batch为单位，按mini-batch进行正规化。具体而言，就是进行使数据分布的均值为0、方差为1的正规化。用数学式表示的话，如下所示。 
$$
\begin{align}
& \mu_B \leftarrow  \frac{1}{m}\sum_{i=1}^mx_i \\
& \sigma^2 \leftarrow  \frac{1}{m}\sum_{i=1}^m(x_i-\mu_B)^2 \\
& \hat{x}_i \leftarrow  \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
\end{align}
$$
这里对mini-batch的m个输入数据的集合$B=\{x_1,\ldots,x_m\}$求均值$\mu_B$。和方差$\sigma_B^2$。 然后，对输入数据进行均值为0、方差为1（合适的分布）的正规化。式中的$\epsilon$是 一个微小值（比如，1e-7等），它是为了防止出现除以0的情况。 通过将这个处理插入到激活函数的前面（或者后面，有文章讨论过应该把Batch Norm加在激活函数前还是后，感兴趣的可以去搜索一下），可以减小数据分布的偏向。

接着，Batch Norm层会对正规化后的数据进行缩放和平移的变换，用数学式表达如下。
$$
y_i \leftarrow \gamma \hat{x}_i + \beta
$$
这里$\gamma$和$\beta$是参数。一开始$\gamma=1, \beta=0$，然后再通过学习调整到合适的值（也就是说这两个参数也要加入神经网络的参数中一起训练）。

![62](https://img.imgdb.cn/item/6031cbe25f4313ce25ed6cc8.png)

Batch Norm的反向传播的推导有些复杂，这里我们不进行介绍。不过如果使用上面的计算图来思考的话，Batch Norm的反向传播或许也能比较轻松地推导出来。Frederik Kratzert的博客[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)里有详细说明，感兴趣的读者可以参考一下（留个坑，之后补上）。 

```python
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 卷积层4维，全连接层2维
        
        # 评估阶段使用的作为数据标准化的均值和方差
        
        self.running_mean = running_mean
        self.running_var = running_var
        
        # backward时使用的中间变量
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
            
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)
    
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
            
            if train_flg:
                mu = x.mean(axis=0)
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + 10e-7)
                xn = xc / std
                
                self.batch_size = x.shape[0]
                self.xc = xc
                self.xn = xn
                self.std = std
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var   
            else:
                xc = x - self.running_mean
                xn = xc / (np.sqrt(self.running_var + 10e-7))
            out = self.gamma * xn + self.beta
            return out
        
        
        def backward(self, dout):
            if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        	dx = self.__backward(dout)

        	dx = dx.reshape(*self.input_shape)
        	return dx
        
        def __backward(self, dout):
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.gamma * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size
            
            self.dgamma = dgamma
            self.dbeta = dbeta
            
            return dx               
```

用MNIST数据集，观察使用Batch Norm层和不用时的区别。下图是权重初始化时不同的尺度参数的结果，实线是使用了Batch Norm，虚线是没有使用。

![63](https://img.imgdb.cn/item/6031cbe25f4313ce25ed6cd5.png)

几乎所有的情况下都是使用Batch Norm时学习进行得更快。同时也发现，实际上，在不使用Batch Norm的情况下，如果不赋予一个尺度好的初始值， 将完全无法进行。 

综上，通过使用Batch Norm，可以推动学习的进行。并且，对权重初始值变得稳健（ “对初始值稳健”表示不那么依赖初始值）。

## 正则化

机器学习的问题中，过拟合是一个很常见的问题。过拟合指的是只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态。机器学习的目标是提高泛化能力，即便是没有包含在练数据里的未观测数据，也希望模型可以进行正确的识别。 我们可以制作复杂的、表现力强的模型，但是相应地，抑制过拟合的技巧也很重要。 

发生过拟合的原因，主要有以下两个。 

- 模型拥有大量参数、表现力强。 

- 训练数据少。 

### 权值衰减（weight decay）

**权值衰减**是一直以来经常被使用的一种抑制过拟合的方法。该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。很多过拟合原木就是因为权重参数取值过大才 发生的。 

复习一下，神经网络的学习目的是减小损失函数的值。这时，例如为损失函数加上权重的平方范数（L2范数）。这样一来，就可以抑制权重变大。用符号表示的话，如果将权重记为W,，L2范数的权值衰减就是$\frac{1}{2}\lambda W^2$，然后将其加到损失函数上。这里， $\lambda$是控制正则化强度的超参数，设置得越大，对大的权重施加的惩罚就越重。

对于所有权重，权值衰减方法都会为损失函数加上$\frac{1}{2}\lambda W^2$。因此，在求权重梯度的计算中，要为之前的误差反向传播法的结果加上正则化项的导数$\lambda W$。 

下图是比较使用和不是权值衰减的MNIST的训练结果。

![64](https://img.imgdb.cn/item/6031cbe25f4313ce25ed6cdd.png)

如图所示，虽然练数据的识别精度和测试数据的识别精度之间有差距，但是与没有使用权值衰减的结果相比，差距变小了。这说明过拟合受到了抑制。此外，还要注意，训练数据的识别精度没有达到100%。

### Dropout

作为抑制过拟合的方法，前面我们介绍了为损失函数加上权重的L2范数的权值衰减方法。该方法可以简单地实现，在某种程度上能够抑制过拟合。但是，如果网络的模型变得很复杂，只用权值衰减就难以应对了。在这种情况下，我们经常会使用Dropout方法。 
Dropout是一种在学习的过程中随机删除神经元的方法。训练时，随机选出隐藏层的神经元，然后将其删除。被删除的神经元不再进行信号的传递。训练时，每传递一次数据，就会随机选择要删除的神经元。然后，测试时，虽然会传递所有的 神经元信号，但是对于各个神经元的输出，要乘上训练时的删除比例后再输出。 

![65](https://img.imgdb.cn/item/6031cbe25f4313ce25ed6ce3.png)

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
```

这里的要点是，每次正向传播时，self. mask中都会以False的形式保存要删除的神经元。self. mask会随机生成和x形状相同的数组，并将值比dropout_ratio大的元素设为True。反向传播时的行为和ReLU相同。也就是说，正向传播时传递了信号的神经元，反向传播时按原样传递信号；正向传播时没有传递信号的神经元，反向传播时信号将停在那里。 

Dropout的实验和前面的实验一样，使用7层网络（每层有100个神经元，激活函数 为ReLU)，左边不使用Dropout，右边使用Dropout，实验的结果如图所示。 

![66](https://img.imgdb.cn/item/6031cc235f4313ce25ed82c8.png)

图中，通过使用Dropout，训练数据和测试数据的识别精度的差距变小了。并且训练数据也没有到达100％的识别精度。像这样，通过使用Dropout，即便是表现力强的网络，也可以抑制过拟合。 

> 机器学习中经常使用集成学习。所谓集成学习，就是让多个模型单独进行学习，推理时再取多个模型的输出的平均值。用神经网络的语境来说，比如，准备5 个结构相同（或者类似）的网络，分别进行学习，测试时，以这5个网络的输出的平 均值作为答案。实验告诉我们，通过进行集成学习，神经网络的识别精度可以提高好几个百分点。这个集成学习与Dropout有密切的关系。这是因为可以将Dropout理解为，通过在学习过程中随机删除神经元，从而每一次都让不同的模型进行学习。推理时，通过对神经元的输出乘以删除比例（比如0.5等），可以取得模型的 均值。也就是说，可以理解成，…**Dropout将集成学习的效果（模拟地）通过一个网络实现了**。 

## 超参数的验证

神经网络中，除了权重和偏置等参数，**超参数**(hyper-parameter)也经常出现。这里所说的超参数是指，比如各层的神经元数量、batch大小、参数更新时的学习率或权值衰减等。如果这些超参数没有设置合适的值，模型的性能就会很差。虽然超参数的取值非 重要，但是在决定超参数的过程中一般会伴随很多的试错。

### 验证数据 

之前我们使用的数据集分成了训练数据和测试数据，训练数据用于学习，测试数据用于评估泛化能力。由此，就可以评估是否只过度拟合了训练数据（是否发生了过拟合）, 以及泛化能力如何等。

下面我们要对超参数设置各种各样的值以进行验证。这里要注意的是，不能使用测试数据评估超参数的性能。这一点非常重要，但也容易被忽视。 为什么不能用测试数据评估超参数的性能呢？这是因为如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合。换句话说，用测试数据确认超参数的值的“好坏”，就会导致超参数的值被调整为只拟合测试数据。这样的话，可能就会得到不能拟合 其他数据、泛化能力低的模型。 因此，调整超参数时，必须使用超参数专用的确认数据。用于调整超参数的数据， 般称为**验证数据**(validation data)。我们使用这个验证数据来评估超参数的好坏。 

> 训练数据用于参数（权重和偏置）的学习，验证数据用于超参数的性能评估。为了确认泛化能力，要在最后使用（比较理想的是只用一次）测试数据。 

### 超参数的最优化

进行超参数的最优化时，逐渐缩小超参数的“好值”的存在范围非常重要。所谓逐渐缩小范围，是指一开始先大致设定一个范围，从这个范围中随机选出一个超参数（采样），用这个采样到的值进行识别精度的评估；然后，多次重复该操作，观察识别精度的结果， 根据这个结果缩小超参数的“好值”的范围。通过重复这一操作，就可以逐渐确定超参数的合适范围。

> 进行神经网络的超参数的最优化时，与网格搜索等有规律的搜索相比，随机采样的搜索方式效果更好。这是因为在多个超参数中，各个超参数对最终的识别精度的影响程度不同。 

超参数的范围只要“大致地指定”就可以了。所谓“大致地指定”，是指像0.001 ($10^{-3}$) 到1000 ($10^3$)这样，以“10的阶乘”的尺度指定范围（也表述为“用对数尺度（log scale) 指定）。 

在超参数的最优化中，要注意的是深度学习需要很长时间（比如，几天或几周）。因此，在超参数的搜索中，需要尽早放弃那些不符合逻辑的超参数。于是，在超参数的最优化中，**减少学习的epoch，缩短一次评估所需的时间是一个不错的办法**。 

## 代码汇总

[Deep_Learning_from_scratch_4](https://github.com/leyuanheart/Deep_Learning_From_Scratch/tree/main/4)

## 补坑：Batch Normalization 反向传播

![89](https://img.imgdb.cn/item/6031d6ea5f4313ce25f16dfe.png)

---

**例行申明**：博客里大部分的文字和图片会使用这本书中的内容，如果是我自己的一些想法，我会说明的（是不是感觉和一般的申明反过来了...，这是我自己的问题，我是这个年三十决定开始写的，但是之前在读的时候并没有做过笔记，现在手边只有这本书，而且我也不想花太多时间在这上面，毕竟还要写论文呢...，但是又不想让之前看过的这本书就这么过去了，所以就决定这么办，如果涉及到版权问题，我到时候就把文章撤了，在这里先道个歉）。

