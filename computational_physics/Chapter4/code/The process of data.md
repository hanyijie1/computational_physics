# The process of data

## 获得数据

## 拟合插值

### 插值

##### 按范围插值

* 内插
* 外插

##### 按插值方法

* 多项式插值：

  * 线性插值：拉格朗日插值法

    随点数增加计算量增加较多，一般与分段插值结合。

    $y(x)=\sum_{j=0}^n A_j(x)y_i$

    $A_j(x)=\prod_{i=0,i\neq j}\frac{x-x_i}{x_j-x_i}$

```python
from scipy import interpolate
interpolate.lagrange(x,y)
#Output: 降序多项式系数

```

* 分段插值

  * 分段拉格朗日插值，缺点是不同区间函数不连续，整体规则会被抑制。
* **样条插值**

  * 三次样条插值：

    target: 二阶导连续，一阶导光滑。

```python
interpolate.splrep(x,y,k=3,s=1.2)
# k=3 拟合次数
# s 代表平滑程度
# return 三元组：节点向量，系数，阶数。

# 调用：
range_x = linspace(1,10,10)
y_interp = interpolate.splev(x, tck)
```

### 曲线拟合

* 判别标准：偏差$\delta=y_{fit}-y_0$
  1. 偏差的绝对值的和最小。
  2. 偏差最大绝对值最小
  3. 偏差平方和最小：**最小二乘法$\min\sum_{t=0}^2\delta$**
* 分类
  1. 基于模型：基于现有物理规则。
  2. 无模型：纯数学，多项式拟合，神经网络，机器学习。
* 拟合函数：

```python
自拟和# 多项式拟合
coeff=np.polyfit(x,y,deg)
#deg: int，the highest coeff
#coeff: return the coeffs of fitting function

# For any self-function
popt,pcov=scipy.optimize.curve_fit(f,xdata,ydata))
#func(x,a,b,c):the form of this function is fixexd.
#popt:the coeffs
#pcov:the error

#最小二乘自拟和函数
scipy.optimize.leastsq(func,x0,args=()))
#func(p,x,y):the form of this function is fixexd.
#x0: the primial values of coeffs

#线性拟合
slope,intercept,rvalue,pvalue,stderr=scipy.stats.linregress(x,y))
#input: x,y
#slope: 斜率
intercept: 截距 

```

### 聚类分析

Acorrding to ?

* Distance: 欧几里和空间距离。
  1. 散点->一级。
  2. 类中心加权。k-means要事先指定分类，且对孤立点敏感。

### 求方差零点

一般以二分法确定位置，其他方法提高精度。

#### 二分法

原理：迭代f(a)<0, f(b)>0中必有一实根。以$\frac{a+b}{2}$作为新区间边界，替换a或b。

问题：多个奇点只得其一，需确定零点区间。

#### 切线法


#### 弦割法


### Fourier transform

#### 离散DFT

$$
w_n=e^{i2\pi/n}\\
x_k=\frac{1}{n}\sum_{m=0}^{n-1}y_mw_n^{-mk}
$$

尼奎斯特频率为采样频率（每秒从信号采样个数）的0.5，采样频率取最高频率的2.54-4倍。

连续信号在采样频率足够高时（带宽两倍），可复刻此连续信号。

#### Fourier series

    任意可积函数可用。

#### 离散时间序列FT

#### FFT

计算次数：O（nlog(n))）<O（n^2）

```python
import scipy.fftpack as spf
spf.fft(x,n=None))
# Input
#x: 元组
#n: 变换长度
```
