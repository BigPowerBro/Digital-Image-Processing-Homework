# 1. Poisson Image Editing

$\qquad$ 泊松图像编辑的目标是自然的融合两个来源不同的图像。对于一般的图像编辑，我们得到的融合图像往往会有
违和的边缘过度。而泊松图像编辑通过将源图像按照原始梯度变化到目标图像的边界，实现自然的融合效果。
$\qquad$ 我们令$S$表示$R^2$的一个闭子集，表示目标图像的定义域。设$\Omega$为$S$的一个以$\partial \Omega$为边界的闭区域。设$f^*$表示定义在$S$去掉$\Omega$内部上的标量函数，$f$表示定义在$\Omega$内部的未知标量函数。最后$\mathbf{v}$表示定义在$\Omega$上的向量场，称为引导向量场。
<img src=pic/fig1.png>
$\qquad$ 为了满足要求：

$\qquad$$\qquad$ (1) $f$的梯度要尽量和引导向量场$\mathbf{v}$接近。
$\qquad$$\qquad$ (2) $f$在$\partial \Omega$上的值要与$f^*$相等。

$\qquad$ 我们得到一个最优化问题：

$$\min_{f} \iint_{\Omega} |\nabla f - \mathbf{v}|^2 d\Omega \quad with \quad f|_{\partial \Omega} = f^*|_{\partial \Omega}$$

$\qquad$转化为迪利克雷边界条件的泊松偏微分方程为

$$\Delta f = div\mathbf{v} \quad over \Omega \quad with \quad f|_{\partial \Omega} = f^*|_{\partial \Omega}$$

$\qquad$引导向量场的选择为源图像的梯度场，即

$$\mathbf{v} = \nabla g$$

$\qquad$于是偏微分方程为

$$\Delta f = \Delta g \quad over \Omega \quad with \quad f|_{\partial \Omega} = f^*|_{\partial \Omega}$$

$\qquad$在实现时我们使用梯度下降法，利用pytorch的卷积函数来计算拉普拉斯距离损失，自动使用梯度下降来降低损失。结果如下图所示：
<div align=center><image src=pic/fig2.png></div><br>
<div align=center><image src=pic/fig3.png></div><br>

# 2. Pix2Pix

$\qquad$ 使用的神经网络为FCN-8s，其结果如下图所示

<div align=center><image src=pic/fig4.png></div><br>

$\qquad$ 对于facedes数据集，跑了140个epoch，其结果如下：

<div align=center><image src=pic/fig5.png></div><br>
<div align=center><image src=pic/fig6.png></div><br>

$\qquad$上面是训练集的结果，下面是验证集的结果。可见在验证集上的结果较差，出现了过拟合的情况。

$\qquad$下面用更大的数据集edges2shoes进行试验，跑了60个epoch，结果如下

<div align=center><image src=pic/fig7.png></div><br>
<div align=center><image src=pic/fig8.png></div><br>

$\qquad$ 此时模型在训练集和验证集上的结果都比较好。