---
layout:     post
title:      Deep learning from scratch 5
subtitle:   Convolution Neural Network
date:       2021-02-20
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Deep Learning
---

<img src="https://img.imgdb.cn/item/60311e2d5f4313ce25b859d6.jpg" alt="fengmian" style="zoom:50%;" />

# 卷积神经网络

本篇的主题是卷积神经网络（Convolutional Neural Network，CNN)。CNN被用于图像识别、语音识别等各种场合，在图像识别的比赛中，基于深度学习的方法几乎都以 CNN为基础。

## 整体结构

之前的神经网络，相邻层的所有神经元之间都有连接。这称为**全连接**（full-connected）。如下图所示，全连接的神经网络中，Affine层后面跟着激活函数ReLU层（或者 Sigmoid层）。这里堆叠了4层“Affine-ReLU'’组合，然后第5层是Affine层，最后由 Softmax层输出最终结果（概率）。 

![67](https://img.imgdb.cn/item/6031cc235f4313ce25ed82d0.png)

那么，CNN又是什么样的结构呢？下图是一个CNN的例子。

![68](https://img.imgdb.cn/item/6031cc235f4313ce25ed82d6.png)

CNN中新增了Convolution层和Pooling层。CNN的层的连接顺序是“Convolution - ReLU -（Pooling)” (Pooling层有时会被省略）。这可以理解为之前 的“Affine - ReLU”连接被替换成了“Convolution - ReLU -（Pooling)”连接。 

还需要注意的是，在图中，靠近输出的层中使用了之前的‘'Affine - ReLU'，组合。此外，最后的输出层中使用了之前的‘'Affine一Softmax”组合。这些都是一般 的CNN中比较常见的结构。 

## 卷积层（Convolution）

### 全连接层存在的问题

全连接层存在什么问题呢？那就是数据的形状被“忽视”了。比如，输入数据是图像时，图像通常是高、长、通道方向上的3维形状。但是，向全连接层输入时，需要将3维数据拉平为1维数据。实际上，前面提到的使用了MNIST数据集的例子中，输入图像就 是1通道、高28像素、长28像素的(1,28,28)形状，但却被排成1列，以784个数据的形式输入到最开始的Affine层。 

图像是3维形状，这个形状中应该含有重要的空间信息。比如，空间上邻近的像素为相似的值、RBG的各个通道之间分别有密切的关联性、相距较远的像素之间没有什么关 联等，3维形状中可能隐藏有值得提取的本质模式。但是，因为全连接层会忽视形状，将 全部的输入数据作为相同的神经元（同一维度的神经元）处理，所以无法利用与形状相关的信息。

而卷积层可以保持形状不变。当输入数据是图像时，卷积层会以3维数据的形式接收输入数据，并同样以3维数据的形式输出至下一层。因此，在CNN中，可以（有可能） 正确理解图像等具有形状的数据。另外，CNN中，有时将卷积层的输入输出数据称为**特征图**（feature map)。

### 卷积运算

卷积层进行的处理就是卷积运算。卷积运算相当于图像处理中的“滤波器运算”。在介绍卷积运算时，我们来看一个具体的例子。 

![69](https://img.imgdb.cn/item/6031cc235f4313ce25ed82e1.png)

如图所示，卷积运算对输入数据应用滤波器。在这个例子中，输入数据是有高长方向的形状的数据，滤波器也一样，有高长方向上的维度。假设用（height, width）表示数据和滤波器的形状，则在本例中，输入大小是（4, 4）滤波器大小是（3, 3），输出大小是 (2, 2)。另外，有的文献中也会用“核”这个词来表示这里所说的“滤波器”。 

现在来解释一下卷积运算的例子中都进行了什么样的计算。下图中展示了卷积运算的计算顺序。

![70](https://img.imgdb.cn/item/6031cc235f4313ce25ed82ec.png) 

对于输入数据，卷积运算以一定的间隔滑动滤波器的窗口并应用。这里所说的创库是指上图中灰色的$3 \times 3$的部分。将各个位置上滤波器的元素和输入的对应元素相乘，然后再求和（有时将这个计算称为**乘积累加运算**）。然后，将这个结果保存到输出的对应位置。将这个过程在所有位置都进行一遍，就可以得到卷积运算的输出。 

在全连接的神经网络中，除了权重参数，还存在偏置。CNN中，滤波器的参数就对 应之前的权重。并且，CNN中也存在偏置。包含偏置的卷积运算的处理流如下图所示。 

![71](https://img.imgdb.cn/item/603206325f4313ce2501dde2.png)

如图所示，向应用了滤波器的数据加上了偏置。偏置通常只有1个（1 x 1)（本例中，相对于应用了滤波器的4个数据，偏置只有1个），这个值会被加到应用了滤波器的所有元素上。 

### 填充（padding）

在进行卷积层的处理之前，有时要向输入数据的周围填入固定的数据（比如0等）, 这称为填充（padding)，是卷积运算中经常会用到的处理。比如，在下图的例子中， 对大小为（4,4)的输入数据应用了幅度为1的填充。“幅度为1的填充”是指用幅度为1像素的0填充周围。 

![72](https://img.imgdb.cn/item/603206325f4313ce2501ddeb.png)

使用填充主要是为了调整输出的大小。比如，对大小为（4, 4）的输入数据 应用（(3, 3)的滤波器时，输出大小变为（2, 2)，相当于输出大小比输入大小缩小了2个元素。这在反复进行多次卷积运算的深度网络中会成为问题。为什么呢？因为如果每次进行卷积运算都会缩小空间，那么在某个时刻输出大小就有可能变为1，导致无法再应用卷积运算。为了避免出现这样的情况，就要使用填充。在刚才的例子中，将 填充的幅度设为1，那么相对于输入大小（4, 4)，输出大小也保持为原来的（4,4)。因 此，卷积运算就可以在保持空间大小不变的情况下将数据传给下一层。 

### 步幅（stride）

应用滤波器的位置的间隔称为**步幅“”（stride）。之前的例子中步幅都为1，如果将步幅设为2，则如下图。

![73](https://img.imgdb.cn/item/603206325f4313ce2501ddf2.png)

综上，增大步幅后，输出大小会变小。而增大填充后，输出大小会变大。这其中是可以总结规律的。

假设输入大小为$(H,W)$，滤波器大小为$(FH,FW)$，输出大小为$(OH,OW)$，填充为$P$，步幅为$S$。此时，


$$
\begin{gather}
OH = \frac{H+2P-FH}{S} + 1 \\
OW = \frac{W+2P-FW}{S} + 1
\end{gather}
$$


这里需要注意的是，虽然只要代入值就可以计算输出的大小，但是所设定的值必须使上式中的分式可以除尽。当输出大小无法除尽（是小数）时，需要采取报错等对策。顺便说一下，根据深度学习的框架不同，当值无法除尽时，有时会向最接近的整数四舍五入、不尽心报错而继续运行。

### 3维数据的卷积计算

之前的卷积运算的例子都是以有高、长方向的2维形状为对象的。但是，图像是3维数据，除了高、长方向之外，还需要处理通道方向。这里，我们按照与之前相同的顺序， 看一下对加上了通道方向的3维数据进行卷积运算的例子。 

下图以3通道的数据为例，展示了卷积运算的结果。和2维数据时相比，可以发现纵深方向（通道方向）上 特征图增加了。通道方向上有多个特征图时，会按通道进行输入数据和滤波器的卷积运算，并将结果相加，从而得到输出。 

![74](https://img.imgdb.cn/item/603206325f4313ce2501ddff.png)

需要注意的是，在三位数据的卷积运算中，输入数据和滤波器的通道数要设为相同的值。

将数据和滤波器结合长方体的方块来考虑，3维数据的卷积运算会很容易理解。方块 是如下图所示的3维长方体。把3维数据表示为多维数组时，书写顺序为（(channel, height, width)。比如，通道数为C、高度为H、长度为W的数据的形状可以写成（C,H, W)。滤波器也一样，要按（channel, height, width）的顺序书写。比如，通道数为C滤波器高度为FH (Filter Height)、长度为FW (Filter Width）时，可以写成（C, FH, FW）。 

![75](https://img.imgdb.cn/item/603206325f4313ce2501de06.png)

在这个例子中，数据输出是1张特征图。所谓1张特征图，换句话说，就是通道数为 1的特征图。那么，如果要在通道方向上也拥有多个卷积运算的输出，该怎么做呢？为此，就需要用到多个滤波器（权重）。用图表示的话，如下。

![76](https://img.imgdb.cn/item/603206675f4313ce2501f4c2.png)

关于偏置的话，每个输出通道只有一个偏置。这里，偏置的形状为$(FN,1,1)$。

![77](https://img.imgdb.cn/item/603206675f4313ce2501f4c8.png)

### 批处理

神经网络的处理中进行了将输入数据打包的批处理。之前的全连接神经网络的实现也 对应了批处理，通过批处理，能够实现处理的高效化和学习时对mini-batch的对应。 

我们希望卷积运算也同样对应批处理。为此，需要将在各层间传递的数据保存为4维数据。具体地讲，就是按(batch_num, channel, height, width）的顺序保存数据。比如，  将上图中的处理改成对N个数据进行批处理时，数据的形状如下图所示。 

批处理版的数据流中，在各个数据的开头添加了批用的维度。像这样，数据作为4维的形状在各层间传递。这里需要注意的是，网络间传递的是4维数据，对这N个数据进行了卷积运算。也就是说，批处理将N次的处理汇总成了1次进行。 

![78](https://img.imgdb.cn/item/603206675f4313ce2501f4d3.png)

## 池化层（Pooling）

池化是缩小高、长方向上的空间的运算。比如，将$2 \times 2$的区域集约成1个元素的处理。

![79](https://img.imgdb.cn/item/603206675f4313ce2501f4de.png)

一般来说，池化的窗口大小会和步幅设定成相同的值。除了Max池化之外，还有Average池化等。相对于Max池化是从目标 区域中取出最大值，Average池化则是计算目标区域的平均值。在图像识别领域，主要使用Max池化。

池化层有一下特征。

- 没有要学习的参数
- 通道数不发生变化：计算是按通道独立进行的。
- 对微小的位置变化具有鲁棒性：输入数据发生微小偏差时，池化仍可以返回相同的结果。

## 卷积层和池化层的实现

如前所述，CNN中各层间传递的数据是4维数据。如果老老实实地实现卷积运算，估计要重复好几层for循环。这样的实现有点麻烦，而且，NumPy中存在使用for语句处理变慢的缺点（NumPy中，访问元素时最好不要用for语句）。这里，我们通过`im2col`（image to column）这个函数来实现。

`im2col`会把输入数据展开以适合滤波器（权重）。具体地说，对于输入数据，将应用滤波器的区域（3维方块）横向展开为1列。

![80](https://img.imgdb.cn/item/603206675f4313ce2501f4ea.png)

在图7-18中，为了便于观察，将步幅设置得很大，以使滤波器的应用区域不重叠。 而在实际的卷积运算中，滤波器的应用区域几乎都是重叠的。在滤波器的应用区域重叠的情况下，使用`im2col`展开后，展开后的元素个数会多于原方块的元素个数。因此，使用 `im2col`的实现存在比普通的实现消耗更多内存的缺点。但是，汇总成一个大的矩阵进行计算，对计算机的计算颇有益处。比如，在矩阵计算的库（线性代数库）等中，矩阵计算的实现己被高度最优化，可以高速地进行大矩阵的乘法运算。因此，通过归结到矩阵计算上，可以有效地利用线性代数库。 

使用 `im2col`展开数据后，只需要将卷积层的滤波器（权重）纵向展开为1列，并计算2个矩阵的乘积即可，如下图。

![81](https://img.imgdb.cn/item/603206ac5f4313ce25020e02.png)

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由（数据量，通道，高，长）的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充
    Returns
    -------
    col : 2维数组
    """
    
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    '''
    上面那段构造col的代码不是很好懂，也可能是我自己的逻辑没理清，我按照自己的逻辑写了下面代码，
    结果已经验证过是一样的了，供参考
    '''
    
    '''
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))'''
    '''
    for y in range(out_h):'''
        '''
        y_start =  stride * y'''
        '''
        y_end = y_start + filter_h'''
        '''
        for x in range(out_w):'''
            '''
            x_start = stride * x'''
            '''
            x_end = x_start + filter_w'''
            '''
            col[:, :, :, :, y, x] = img[:, :, y_start:y_end, x_start:x_end]'''
    
            
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)    # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)   # (90, 75)
```

> 值得一提的是，即使已经使用了这样的技巧，但是卷积操作依然是相当耗费时间的。在AlexNet中，卷积层的处理时间加起来占GPU整体的95%，占CPU整体的89%。

### 卷积层的实现

先写一个不考虑反向传播的实现。

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad) # [N*out_h*out_w, C*FH*FW]
        
        col_W = self.W.reshape(FN, -1).T    # [C*FH*FW, FN]
        
        out = np.dot(col, col_W) + self.b  # [N*out_h*out_w, FN]
        
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        return out
```

卷积层的初始化方法将滤波器（权重）、偏置、步幅、填充作为参数接收。滤波器是 （FN, C, FH, FW）的4维形状，分别是Filter Number（滤波器数量）、Channel、 Filter Height、 Filter Width的缩写。 

展开滤波器的部分将各个滤波器的方块纵向 展开为1列。这里通过。reshape(FN, -1）将参数指定为-1，这是reshape的一个便利的功能。通过在reshape时指定为-1，reshape函数会自动计算-1维度上的元素个数， 以使多维数组的元索个数前后一致。比如，(10, 3, 5, 5)形状的数组的元素个数共有750 个，指定reshape(10，-1）后，就会转换成（10, 75）形状的数组。 

forward的实现中，最后会将输出大小转换为合适的形状。转换时使用了NumPy的 transpose函数。transpose会更改多维数组的轴的顺序。如下图所示，通过指定从 0开始的索引（编号）序列，就可以更改轴的顺序。 

![82](https://img.imgdb.cn/item/603206ac5f4313ce25020e0f.png)

以上就是卷积层的forward处理的实现。通过使用`im2col`进行展开，基本上可以 像实现全连接层的Affine层一样来实现。接下来是卷积层的反向传播的实现， 因为和Affine层的实现有很多共通的地方，所以就不再介绍了。但有一点需要注意，在进行卷积层的反向传播时，必须进行`im2col`的逆处理。

```python
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 输入的形状（例：(10, 1, 28, 28)）
    filter_h
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    '''
    和img2col同样的，我也没能理解上面的逻辑，还是按自己的逻辑写了一个 '''
    '''
    img = np.zeros((N, C, stride * (out_h - 1) + filter_h + 1, stride * (out_w - 1) + filter_w + 1))'''
    
    # +1是为了防止出现正向传播时没法整除的情况。不过如果出现了不整除的情况，就已经无法再还原成原来一模一样了，可以自己试试
    
    '''
    for y in range(out_h):'''
        '''
        y_start =  stride * y'''
        '''
        y_end = y_start + filter_h'''
        '''
        for x in range(out_w):'''
            '''
            x_start = stride * x'''
            '''
            x_end = x_start + filter_w'''
            '''
            img[:, :, y_start:y_end, x_start:x_end] = col[:, :, :, :, y, x]'''


    return img[:, :, pad:H + pad, pad:W + pad]
```

现在可以实现完整的Convolution层了。

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        #  backward时要用到的中间变量
        
        self.x = None
        self.col = None
        self.col_W = None
        
        # 参数的梯度
        
        self.dW = None
        self.db = None
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad) # [N*out_h*out_w, C*FH*FW]
        
        col_W = self.W.reshape(FN, -1).T    # [C*FH*FW, FN]
        
        out = np.dot(col, col_W) + self.b  # [N*out_h*out_w, FN]
        
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col  
        self.col_W = col_W
        
        return out
    
    def backward(self, dout): # dout:[N, FN, out_h, out_w]
        
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) #[N*out_h*out_w, FN]
        
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout) # [C*Fh*FW, FN]
        
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)  # [N*out_h*out_w, C*Fh*FW]
        
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        return dx
```

说实话，这个程序应该是本书中最难写的了，困难主要在矩阵shape的对应上，很容易就晕了。

### 池化层的实现

池化层的实现和卷积层相同，都是用`img2col`展开数据。不过，展开在通道方向上是独立的，这一点和卷积层不同。

![83](https://img.imgdb.cn/item/603206ac5f4313ce25020e14.png)

按照上图的流程，就可以写出池化层的forward处理。关于backward处理，可以参考ReLU层的实现中使用的max的反向传播。

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        elf.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
        
    def foward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)   # [N*out_h*out_w, C*FH*FW]
        
        col = col.reshape(-1, self.pool_h*self.pool_w) # 按通道展开[N*out_h*out_w*C, FH*FW]
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        self.x = x
        self.arg_max = arg_max
        return out
    
    
    def backward(self, dout): # dout:[Nout, C, out_h, out_w]
        
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))  # [Nout*C*out_h*out_w, FH*FW]
        
        dmax[np.arange(self.arange(self.arg_max)), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, )) # [Nout, out_h, out_w, C, FH*FW]
        
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)  # [Nout*out_h*out_w, C*FH*FW]
        
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
```

## CNN的实现

有了之前的卷积层和池化层，我们就可以来组合这些层，搭建CNN了。

![84](https://img.imgdb.cn/item/603206ac5f4313ce25020e19.png)

如图，网络的构成是“Convolution - ReLU - Pooling - Affine - ReLU - Affine - Softmax”，我们将它实现名为SimpleConvNet的类（没有考虑批处理）。

首先来看一下SimpleConvNet的初始化（`__init__` )，取下面这些参数参数 。

- input_ dim ——输入数据的维度：（通道，高，长） 
- conv_param ——卷积层的超参数（字典）。字典的关键字如下：
- - filter_num ——滤波器的数量 
- - filter_size ——滤波器的大小 
- - stride ——步幅 
- - pad——填充 
- hidden size ——隐藏层（全连接）的神经元数量 
- output_size ——输出层（全连接）的神经元数量 
- weitght_int_std ——初始化时权重的标准差

这里，卷积层的超参数通过名为conv_param的字典传入。 比如{}这样，保存必要的超参数。

SimpleConvNet的初始化的实现稍长，我们分成3部分来说明，首先是初始化的最 开始部分。 

```python
class SimpleConvNet:
    """简单ConvNet
    conv - relu - pool - affine - relu - affine - softmax
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
```

这里将由初始化参数传入的卷积层的超参数从字典中取了出来（以方便后面使用）, 然后，计算卷积层的输出大小。接下来是权重参数的初始化部分。

```python
self.params = {}
self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
self.params['b1'] = np.zeros(filter_num)
self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
self.params['b2'] = np.zeros(hidden_size)
self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
self.params['b3'] = np.zeros(output_size)
```

最后，生成必要的层。

```python
self.layers = OrderedDict()
self.layers['Conv1'] = Convolution(self.params['W1'], 
                                   self.params['b1'], 
                                   conv_param['stride'], 
                                   conv_param['pad'])
self.layers['Relu1'] = Relu()
self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
self.layers['Relu2'] = Relu()
self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
self.last_layer = SoftmaxWithLoss()
```

以上就是SimpleConvNet的初始化中进行的处理。像这样初始化后，进行推理的 predict方法和求损失函数值的loss方法就可以像下面这样实现。 

```python
def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

def loss(self, x, t):
    y = self.predict(x)
    return self.last_layer.forward(y, t)
```

接下来是基于误差反向传播法求梯度的代码实现。

```python
def gradient(self, x, t):
        # forward
        
        self.loss(x, t)

        # backward
        
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```

## CNN的可视化

CNN中用到的卷积层在“观察”什么呢？本节将通过卷积层的可视化，探索CNN中到底进行了什么处理。 

### 第1层权重的可视化 

第1层的卷积层的权重 的形状是（30, 1, 5, 5），即30个大小为$5 \times 5$、通道为1的滤波器。滤波器大小是$5 \times 5$，通道数是1，意味着滤波器可以可视化为1通道的灰度图像。现在，我们将卷积层（第1 层）的滤波器显示为图像。这里，我们来比较一下学习前和学习后的权重，结果如下图所示。学习前的滤波器是随机进行初始化的，所以在黑白的浓淡上没有规律可 循，但学习后的滤波器变成了有规律的图像。我们发现，通过学习，滤波器被更新成了有规律的滤波器，比如从白到黑渐变的滤波器、含有块状区域（称为blob）的滤波器等。 

![85](https://img.imgdb.cn/item/603206ac5f4313ce25020e21.png)

如果要问上图中右边的有规律的滤波器在“观察”什么，答案就是它在观察边缘 （颜色变化的分界线）和斑块（局部的块状区域）等。比如，左半部分为白色、右半部分为黑色的滤波器的情况下，如下图所示，会对垂直方向上的边缘有响应。 

![86](https://img.imgdb.cn/item/603206f15f4313ce2502284f.png)

由此可知，刚才实现的卷积层的滤波器会提取边缘或者斑块等原始信息。

## 加深层的动机

这是最后一章里的一个小节。最后一章主要介绍了深层网络有关的内容，比如深度学习的小历史、深度学习的高速化以及深度学习的应用案例，介绍得比较泛，不过这一部分的内容我觉得对于理解为什么要加深层有帮助，所以介绍一下。

关于加深层的重要性，现状是理论研究还不够透彻。尽管目前相关理论还比较贫乏， 但是有几点可以从过往的研究和实验中得以解释（虽然有一些直观）。本节就加深层的重要性，给出一些增补性的数据和说明。

下面我们说一下加深层的好处。其中一个好处就是可以减少网络的参数数量。说得详细一点，就是与没有加深层的网络相比，加深了层的网络可以用更少的参数达到同等水平 （或者更强）的表现力。这一点结合卷积运算中的滤波器大小来思考就好理解了。比如， 下图展示了由$5 \times 5$的滤波器构成的卷积层。

![87](https://img.imgdb.cn/item/603206f15f4313ce25022859.png)

这里希望大家考虑一下输出数据的各个节点是**从输入数据的哪个区域**计算出来的。显然，在上图的例子中，每个输出节点都是从输入数据的某个5x5的区域算出来的。接 下来我们思考一下下图中重复两次3x3的卷积运算的情形。此时，每个输出节点将由中间数据的某个3x3的区域计算出来。那么，中间数据的3x3的区域又是由前一个输 入数据的哪个区域计算出来的呢？仔细观察下图，可知它对应一个5x 5的区域。也就 是说，下图的输出数据是“观察”了输入数据的某个5x5的区域后计算出来的。 

![88](https://img.imgdb.cn/item/603206f15f4313ce25022861.png)

一次5x5的卷积运算的区域可以由两次3x3的卷积运算代替。并且，相对于前者的参数数量25 (5x5)，后者一共是18 (2x3x3)，通过叠加卷积层，参数数量减少了。而且，这个参数数量之差会随着层的加深而变大。比如，重复三次3x3的卷积运算，参数的数量总共是27。而为了用一次卷积运算“观察”与之相同的区域，需要一个7x 7的滤波器，此时的参数数量是490。

> 叠加小型滤波器来加深网络的好处是可以减少参数的数量，扩大感受野 (receptive field，给神经元施加变化的某个局部空间区域，感受野是一个神经元对原始图像的连接。通常说：第几层对输入数据（即原始图像）的感受野）。并且，通过叠加层，将 ReLU等激活函数夹在卷积层的中间，进一步提高了网络的表现力。这是因为向网络 添加了基于激活函数的“非线性”表现力，通过非线性函数的叠加，可以表现更加复杂的东西。 

加深层的另一个好处就是使学习更加高效。与没有加深层的网络相比，通过加深层， 可以减少学习数据，从而高效地进行学习。为了直观地理解这一点，之前介绍了CNN的卷积层会分层次地提取信息。具体地说，在前面的卷积层中，神经元会对边缘等简单的形状有响应，随着层的加深，开始对纹理、物体部件等更加复杂的东西有响应。 

我们考虑一下“狗”的识别问题。要用浅层网络解决 这个问题的话，卷积层需要一下子理解很多“狗”的特征。“狗”有各种各样的种类，根据拍摄环境的不同，外观变化也很大。因此，要理解“狗”的特征，需要大量富有差异性的学习数据，而这会导致学习需要花费很多时间。 

不过，通过加深网络，就可以分层次地分解需要学习的问题。因此，各层需要学习的问题就变成了更简单的问题。比如，最开始的层只要专注于学习边缘就好，这样一来，只需用较少的学习数据就可以高效地进行学习。这是为什么呢？因为和印有“狗”的照片相比，包含边缘的图像数量众多，并且边缘的模式比“狗”的模式结构更简单。 

通过加深层，可以分层次地传递信息，这一点也很重要。比如，因为提取了边缘的层的下一层能够使用边缘的信息，所以应该能够高效地学习更加高级的模式。也就是说，通过加深层，可以将各层要学习的问题分解成容易解决的简单问题，从而可以进行高效的学 习。

以上就是对加深层的重要性的增补性说明。不过，这里需要注意的是，近几年的深层化是由大数据、计算能力等即便加深层也能正确地进行学习的新技术和环境支撑的。 

## 代码汇总

[Deep_Learning_from_scratch_5](https://github.com/leyuanheart/Deep_Learning_From_Scratch/tree/main/5)




---

花了整整8天时间，把这本书总结完了，还剩下代码的整理工作（应该要也要花上1天），终于，完成一个计划。

**例行申明**：博客里大部分的文字和图片会使用这本书中的内容，如果是我自己的一些想法，我会说明的（是不是感觉和一般的申明反过来了...，这是我自己的问题，我是这个年三十决定开始写的，但是之前在读的时候并没有做过笔记，现在手边只有这本书，而且我也不想花太多时间在这上面，毕竟还要写论文呢...，但是又不想让之前看过的这本书就这么过去了，所以就决定这么办，如果涉及到版权问题，我到时候就把文章撤了，在这里先道个歉）。

