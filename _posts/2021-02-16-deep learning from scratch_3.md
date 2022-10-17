---
layout:     post
title:      Deep learning from scratch 3
subtitle:   Error Backward-Propagation Method
date:       2021-02-16
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Deep Learning
---

<img src="https://img.imgdb.cn/item/60311e2d5f4313ce25b859d6.jpg" alt="fengmian" style="zoom:50%;" />

## 误差反向传播法

上一篇中，我们介绍了神经网络的学习，并通过数值微分计算了神经网络的权重参数的梯度（严格来说，是损失函数关于权重参数的梯度）。数值微分虽然简单，也容易实现，但缺点是计算上比较费时间。本篇我们将学习一个能够高效计算权重参数的梯度的方法一误差反向传播法。

关于这个反向传播算法，我之前接触深度学习的时候，就一直把它看成链式求导法则在神经网络中的应用（我现在其实也还是这么认为，取一个这个看起来“高大上”的名字...）。作者也说了，以他个人的观点，有两种方法去理解反向传播：一种是基于数学式（也就是链式法则）；另一种是基于**计算图**（computation graph）。虽然我的感觉计算图也只是让我们能更直观地理解，核心还是链式法则，但是我同样也觉得有必要介绍这种方法（理由说不太清，反正就是想）。

### 计算图

计算图将计算过程用图形表示出来。这里说的图形是数据结构图，通过多个节点和边表示（连接节点的直线称为“边”）。为了让大家熟悉计算图，先用计算图解一些简单的问题。从这些简单的问题开始，逐步深入，最终抵达误差反向传播法。

问题：太郎在超市买了2 个苹果、3 个橘子。其中，苹果每个 100 日元，橘子每个 150 日元。消费税是 10%，请计算支付金额。

![30](https://img.imgdb.cn/item/6031c1d35f4313ce25e9cf15.png)

计算图通过节点和箭头表示计算过程。节点用$\bigcirc$表示，$\bigcirc$中是要进行的操作。将计算的中间结果写在箭头的上方，表示各个节点的计算结果从左向右传递。

#### 局部计算

计算图的特征是可以通过传递“局部计算"获得最终结果。“局部”这个词的意思是“与 自己相关的某个小范围”。局部计算是指，无论全局发生了什么，都能只根据与自己相关的信息输出接下来的结果。

我们用一个具体的例子来说明局部计算。比如，在超市买了 2 个苹果和其他很多东西。此时，可以画出如下的计算图。

![31](https://img.imgdb.cn/item/6031c5d35f4313ce25eb38a1.png)

假设（经过复杂的计算）购买的其他很多东西总共花费 4000 日元。 这里的重点是，各个节点处的计算都是局部计算。这意味着，例如苹果和其他很多东西的求和运算 $(4000+200 \rightarrow 4200)$ 并不关心 4000 这个数字是如何计算而来的，只要把两个数字相加就可以了。换言之，各个节点处只需进行与自己有关的计算（在这个例子中是对输入的两个数字进行加法运算），不用考虑全局。

综上，计算图可以集中精力于局部计算。无论全局的计算有多么复杂，各个步骤所要做的就是对象节点的局部计算。虽然局部计算非常简单，但是通过传递它的计算结果，可以获得全局的复杂计算的结果。

#### 为何用计算图解题

第一个就是前面所说的局部计算。无论全局是多么复杂的计算，都可以通过局部计算使各个节点致力于简单的计算，从而简化问题。另一个优点是，利用计算图可以将中间的计算结果全部保存起来（比如，计算进行到 2 个苹果时的金额是 200 日元、加上消费税之前的金额 650 日元等）。这为之后高效地进行反向传播提供了基础。

#### 计算图的反向传播

前面介绍的计算图的正向传播将计算结果正向（从左到右）传递，其计算过程是我们日常接触的计算过程，所以感觉上可能比较自然。而反向传播将局部导数向正方向的反方向（从右到左）传递，一开始可能会让人感到困惑。传递这个局部导数的原理，是基于链式法则（chain rule）的。

假设存在$y=f(x)$的计算，这个计算的反向传播如下图：

![32](https://img.imgdb.cn/item/6031c5d35f4313ce25eb38a4.png)

如图所示，反向传播的计算顺序是，将信号 $E$ 乘以节点的局部导数 $\left(\frac{\partial y}{\partial x}\right),$ 然后将结果传递给下一个节点。这里所说的局部导数是指正向传播中 $y=f(x)$ 的导数，也就是 $y$ 关于 $x$ 的导数 $\left(\frac{\partial y}{\partial x}\right)$。 比如，假设 $y=f(x)=x^{2},$ 则局部导数为 $\frac{\partial y}{\partial x}=2 x$ 。把这个局部导数乘以上游传过来的值 (本例中为 $E$ ），然后传递给前面的节点。

这就是反向传播的计算顺序。通过这样的计算，可以高效地求出导数的值，这是反向传播的要点。

### 链式法则和计算图

求导链式法则我就不介绍了，直接看一下它是如何在计算图中使用的。

我们以一个函数为例：$z=(x+y)^2$，它可以看成两个式子的复合：


$$
\begin{align}
&z = t^2\\
&t = x + y
\end{align}
$$


![33](https://img.imgdb.cn/item/6031c5d35f4313ce25eb38a6.png)

如图所示，计算图的反向传播从右到左传播信号。反向传播的计算顺序是，先将节点的输入信号乘以节点的局部导数（偏导数），然后再传递给下一个节点。比如，反向传播时，“**2”节点输入的是 $\frac{\partial z}{\partial z}$，将其乘以局部导数 $\frac{\partial z}{\partial t}$（因为正向传播时输入是t、输出是z，所以这个节点的局部导数是 $\frac{\partial z}{\partial t}$），然后传递给下一个节点。

上图需要注意的是最左边的反向传播的结果。根据链式法则， $\frac{\partial z}{\partial x}= \frac{\partial z}{\partial z} \frac{\partial z}{\partial t}\frac{\partial t}{\partial x}$ 成立，对应“ $z$ 关于 $x$ 的导数”。也就是说，反向传播是基于链式法则的。

### 反向传播

上一节介绍了计算图的反向传播是基于链式法则成立的。本节将以“+”和“x”等运算为 例，介绍反向传播的结构。

#### 加法节点的反向传播

首先来考虑加法节点的反向传播。这里以 $z=x+y$ 为对象，观察它的反向传播。 $z=x$ $+y$ 的导数可由下式（解析性地）计算出来。


$$
\begin{array}{l}

\frac{\partial z}{\partial x}=1 \\

\frac{\partial z}{\partial y}=1

\end{array}
$$


因此，用计算图表示的话，如下图：

![34](https://img.imgdb.cn/item/6031c5d35f4313ce25eb38aa.png)

在图中，反向传播将从上游传过来的导数（本例中是 $\frac{\partial L}{\partial x}$ ) 乘以 1 , 然后传向下游。也就是说，因为加法节点的反向传播只乘以 1 , 所以输入的值会原封不动地流向下一 个节点。

#### 乘法节点的反向传播

接下来，我们看一下乘法节点的反向传播。这里我们考虑 $z=x y$ 。这个式子的导数表示如下：


$$
\begin{array}{l}

\frac{\partial z}{\partial x}=y \\

\frac{\partial z}{\partial y}=x

\end{array}
$$


根据式子，我们可以画出计算图。

![35](https://img.imgdb.cn/item/6031c5d35f4313ce25eb38ac.png)

乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”后传递给下游。 翻转值表示一种翻转关系，如图 正向传播时信号是 $x$ 的话，反向传播时则是 $y$； 正向传播时信号是 $y$ 的话，反向传播时则是 $x$。另外，加法的反向传播知识将上游的值传给下游，并不需要正向传播的输入信号。但是，**乘法的反向传播需要正向传播时的输入信号值**。因此，实现乘法节点的反向传播时，要保存正向传播的输入信号。

### 简单层的实现

本节将用Python实现前面的购买苹果和桔子的例子。这里，我们把要实现的计算图的乘法节点称为“乘法层”（MulLayer），加法节点称为“加法层”（AddLayer）。

> 在后面我们将把构建神经网络的“层”实现为一个类。这里所说的“层”是神经网络中中功能的单位。之后，我们会把Sigmoid函数、负责矩阵乘积的Affine函数等，都以层为单位进行实现。

#### 乘法层的实现

层的实现中有两个共通的方法（接口）`forward`()和`backward`()。分别对应正向传播和反向传播，像Pytorch这样的深度学习框架也会沿用这样的命名规则。

```python
class MulLayer(object):
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y  # 翻转x和y
        
        dy = dout * self.x
        
        return dx, dy
```

现在我们使用`MulLayer`来实现下图买苹果的例子。

![36](https://img.imgdb.cn/item/6031c60d5f4313ce25eb570a.png)

```python
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))            # 220

print("dApple:", dapple)               # 2.2

print("dApple_num:", int(dapple_num))  # 110

print("dTax:", dtax)                   # 200
```

调用`backward`()的顺序与调用`forward`()的顺序相反。此外，要注意`backward`()的参数中需要输入“关于正向传播时的输出变量的导数”。比如，`mul_apple_layer`在正向传播时会输出apple_price，在反向传播时，则会将apple_price的导数dapple_price设为参数。

#### 加法层的实现

```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
```

然后我们就可以用加法层和乘法层来实现买苹果和桔子的例子了。

![37](https://img.imgdb.cn/item/6031c60d5f4313ce25eb570d.png)

```python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)

orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)

all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)

price = mul_tax_layer.forward(all_price, tax)  # (4)

# backward

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)

dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)

dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)

dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price))             # 715

print("dApple:", dapple)                # 110

print("dApple_num:", int(dapple_num))   # 2.2

print("dOrange:", dorange)              # 3.3

print("dOrange_num:", int(dorange_num)) # 165

print("dTax:", dtax)                    # 650
```

### 激活函数层的实现

#### ReLU层

ReLU（Rectified Linear Unit）由下式表示：


$$
y = \begin{cases}
x, & x >0 \\
0, & x \leq 0
\end{cases}
$$


其导数为：
$$
\frac{\partial y}{\partial x} = \begin{cases}
1, & x >0 \\
0, & x \leq 0
\end{cases}
$$


其实ReLU在0点不可导，而是存在次梯度，在[0, 1]之间。我们可以选择0作为它在0点的“导数”，在这里对结果并不会造成什么影响。

![38](https://img.imgdb.cn/item/6031c60d5f4313ce25eb570f.png)

```python
class Relu(object):
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```

#### Sigmoid层

sigmoid函数的表达式：


$$
y = \frac{1}{1+\exp(-x)}
$$


用计算图表示的话，则为：

![39](https://img.imgdb.cn/item/6031c60d5f4313ce25eb5711.png)

图中，除了“$\times$”和“+”节点之外，还出现了新的“exp”和“/”节点。其实在这里就有些体现出基于数学式的理解和基于计算图理解之间的差异了，如果是基于数学式的话，在这里就直接对sigmoid函数求导了。但是，基于计算图的话，会将这个过程拆解成一些基本的运算步骤，然后一步步的操作。我的感觉是：第一遍学要从计算图学起，得知道是怎么一步步过来的。走完一遍之后，再用的时候其实还是直接求导了。

这一部分比较长，我就直接放书中的截图了。

![40](https://img.imgdb.cn/item/6031c60d5f4313ce25eb5713.png)

```python
class Sigmoid(object):
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
```

### Affine层的实现

几何中，仿射变化包括一次线性变换和一次平移，分别对应神经网络的加权运算和加偏置运算。

![41](https://img.imgdb.cn/item/6031c6395f4313ce25eb6877.png)

要注意这里的$X,W,B$是矩阵（对维数组）。之前计算图中各节点间流动的是标量。相应导数的推导和标量是类似的。“+”对应标量的加法，“dot”对应标量的“$\times$”。只不过变成矩阵后要考虑矩阵的形状。我们知道$W$和$\frac{\partial L}{\partial W}$形状相同。比如，$\frac{\partial L}{\partial Y}$的形状是 (3, )，$W$的形状是 (2, 3)，思考$\frac{\partial L}{\partial Y}$和$W$如何相乘，能使得$\frac{\partial L}{\partial X}$的形状为 (2, )，这样一来，就会自然而然地推导出上面的公式。

书上的这个公式我觉得是有些问题的，它的一些向量的维数没有说得很清楚，比如$\frac{\partial L}{\partial Y}$一下是 (3, )，一下又是 (1, 3)。这里我明确一下，假设$X$的维数是 (1, 2)，1表示一个样本，2表示变量的个数，$W$是(2, 3)，$B$是(1, 3)，那么$Y$就是(1, 3)，那么此时对应的导数公式应该是：


$$
\begin{align}
\frac{\partial L}{\partial X}(1, 2) = \frac{\partial L}{\partial Y}(1, 3) \cdot W^T(3, 2)\\
\frac{\partial L}{\partial W}(2, 3) = X^T(2, 1) \cdot \frac{\partial L}{\partial Y}(1, 3) 
\end{align}
$$


**批版本的Affine层**

与刚刚不同，现在输入$X$的形状是 (N, 2)。之后就和前面一样导出$\frac{\partial L}{\partial X}$和$\frac{\partial L}{\partial W}$，但是对于偏置$B$，需要特别注意，正向传播时，偏置会被加到每一个数据上。因此，反向传播时，各数据的反向传播值需要汇总为偏置的元素。所以，需要用`np.sum`对第0轴（以数据为单位的轴）方向上的元素进行求和。

![42](https://img.imgdb.cn/item/6031c6395f4313ce25eb6879.png)

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
       
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx     
# 这里需要注意一下，返回的是对x的导数，这是因为x在整个网络里流动的，而W和b仅仅是该层具有的参数，不需要往回传了
```

### Softmax-with-Loss层实现

![43](https://img.imgdb.cn/item/6031c6395f4313ce25eb687b.png)

Softmax层将输入值正规化（和调整为1）之后再输出。下面来实现softmax层，因为已经到输出层，所以一并把计算loss的过程也包括进去，统称为“Softmax-with-Loss”层。

![44](https://img.imgdb.cn/item/6031c6395f4313ce25eb687e.png)

这应该是全书看上去最复杂的一个图，它还有一个简易版的图，如下，

![45](https://img.imgdb.cn/item/6031c6395f4313ce25eb6882.png)

全书唯一的附录也是为了推导最后反向传播的及其”漂亮“的结果($y_1-t_1, y_2-t_2, y_3-t_3$)。$y$是softmax的输出，$t$是数据标签。我也会把推导放在最后。这里需要说明一下的是，这样“漂亮”的结果并不是偶然，而是为了得到这样的结果，在输出层使用softmax函数后，特意设计了交叉熵误差函数（这就说得很牛了）。回归问题中输出层使用“恒等函数”，损失函数使用“平方和误差”也同样的理由。不过我持保留意见。

看到这里我的一个收获就是终于明白了为什么要叫做“误差反向传播”。原因就是，本来就是一个通过梯度下降来学习的过程，但是由于最后一层计算出来的梯度恰好是我们通常意义上的“误差”，所以就把链式法则里梯度的回传叫做了误差反向传播。

有了数学式之后，实现起来就很简单了。不过要注意的是，反向传播时，将要传播的值除以batch_size，传递给前面成的是单个数据的平均误差。

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax输出
        
        self.t = None # 标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)   # 之前已经定义过了
        
        self.loss = cross_entropy_error(self.y, self.t)  # 之前已经定义过了
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # one-hot coding 
            
            dx = (self.y - self.t) / batch_size
        else:   # non one-hot coding
            
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
```

### 改写TwoLayerNet

最后我们来个完整的，将上一篇中的TwoLayerNet给更新一下，变动的部分主要有两块：一是将原来计算梯度的数值微分替换成反向传播（其实就是用解析解替换了数值解...）；二是在构建网络的时候使用了**层**。

| 变量                                                    | 说明                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| `params`                                                | 保存神经网络的参数的字典型变量（实例变量）。<br />params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。<br />params['W2'] 是第 2 层的权重，params['b2'] 是第 2 层的偏置。 |
| `layers`                                                | 保存神经网络的层的**有序字典变量**<br />以layers['Affine1']、layers['Relu1']、layers['Affine2']的形式，通过有序字典保存各层 |
| `lastLayer`                                             | 神经网络的最后一层。本例中为SoftmaxWithLoss层                |
|                                                         |                                                              |
|                                                         |                                                              |
| **方法**                                                | **说明**                                                     |
| `__init__ `(self, input_size, hidden_size, output_size) | 进行初始化。<br />参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数 |
| `predict`(self, x)                                      | 进行识别 (推理)。<br />参数 x 是图像数据                     |
| `loss`(self, x, t)                                      | 计算损失函数的值。<br />参数 x 是图像数据，t 是正确解标签（后面 3 个方法的参数也一样） |
| `accuracy`(self, x, t)                                  | 计算识别精度                                                 |
| `numerical_gradient`(self, x, t)                        | 通过数值微分计算关于权重参数的梯度（同上一篇一样）           |
| `gradient`(self, x, t)                                  | 通过误差反向传播法计算关于权重参数的梯度                     |

```python
from collections import OrderedDict

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        # 生成层
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    # x: 输入数据，t: 标签
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        # forward
        
        self.loss(x, t)
        
        # backward
        
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].dW
        
        return grads

```

代码中有些地方要强调一下，就是将神经网络的层保存为`OrderedDict`这一点。`OrderedDict`是有序字典，“有序”是指它可以记住向字典里添加元素的顺序。因此神经网络的正向传播只需要按照添加元素的顺序调用各层的`forward()`方法，而反向传播只需要按照相反的顺序调用各个层即可（当然如果你不需要为每个层命名的话，考虑列表也是可以的，但是层多的时候，仅靠数字索引是很难找到指定的层的，你也记不住...）。所以这里要做的事仅仅是以正确的顺序链接各层，再按顺序（或者逆序）调用各层。

像这样通过将神经网络的组成元素以层的方式实现。可以轻松地苟江神经网络。因为想另外构建一个神经网络（比如5层，10层，20层...的大网络）时，只需要像组装乐高积木一样添加必要的层就可以了。

#### 梯度确认（gradient checking）

到目前为止，我们介绍了两种求梯度的方法。一种是数值微分，一种是解析性地求解数学式。后一种方法通过误差反向传播，即使存在大量的参数，也可以高效地计算梯度。既然后者更高效，那数值微分还有什么用呢？实际上，在确认反向传播法的实现是否正确时，是需要用到数值微分的。

数值微分的优点是实现简单，因此，一般情况下不太容易出错。而误差反向传播的实现很复杂，容易出错。所以，经常会比较数值微分的结果和误差反向传播法的结果，以确认误差反向传播法的实现是否正确。这个操作就称为**梯度确认**（gradient checking）。

我们可以用我们写的TwoLayerNet来试验一下：

```python
# 生成数据

x = np.random.randn(3, 784)
t = np.random.randint(0, 10, (3, ))

# 创建网络

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

grad_numerical = network.numerical_gradient(x, t)
grad_backprop = network.gradient(x, t)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))
    
    
    
# W1:2.5925165667177406e-09

# b1:2.845791808914243e-09

# W2:1.9350786973215555e-08

# b2:1.4160759305298364e-07
```

这里的误差使用的是对应元素差的绝对值。可以看到结果都是接近于0的，说明我们的实现是正确的。

最后在放一张使用本篇构造的网络训练MNIST的结果，代码和上篇几乎一样，只需要把`grad = network.numerical_gradient(x_batch, t_batch)`换成`grad = network.gradient(x_batch, t_batch)`就行了。

![46](https://img.imgdb.cn/item/6031c6745f4313ce25eb8568.png)

## Softmax-with-Loss层的计算图

![47](https://img.imgdb.cn/item/6031c6745f4313ce25eb856a.png)

## 代码汇总

[Deep_Learning_from_scratch_3](https://github.com/leyuanheart/Deep_Learning_From_Scratch/tree/main/3)

---

**例行申明**：博客里大部分的文字和图片会使用这本书中的内容，如果是我自己的一些想法，我会说明的（是不是感觉和一般的申明反过来了...，这是我自己的问题，我是这个年三十决定开始写的，但是之前在读的时候并没有做过笔记，现在手边只有这本书，而且我也不想花太多时间在这上面，毕竟还要写论文呢...，但是又不想让之前看过的这本书就这么过去了，所以就决定这么办，如果涉及到版权问题，我到时候就把文章撤了，在这里先道个歉）。

