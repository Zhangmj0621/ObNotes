RoPE的全称是旋转式位置编码（Rotary Position Embedding，RoPE），这是一种配合Attention机制能达到“绝对位置编码的方式实现相对位置编码”的设计，本文从数学上证明RoPE的原理；

[https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)

# 1. 基本思路

为了达到这个目的，我们假设通过下述运算来给q,k添加绝对位置信息：

$q_m=f(q,m),k_n=f(k,n)$

也就是说，我们分别为q,k设计操作 $f(⋅,m),f(⋅,n)$，使得经过该操作后， $q_m, k_n$就带有了位置m,n的绝对位置信息。Attention的核心运算是内积，所以我们希望的内积的结果带有相对位置信息，因此假设存在恒等关系： $⟨f(q,m),f(k,n)⟩=g(q,k,m−n)$ 所以我们要求出该恒等式的一个（尽可能简单的）解。求解过程还需要一些初始条件，显然我们可以合理地设 $f(q,0)=q和f(k,0)=k$。

# 2. 求解过程

在复数中有 $\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$，$\text{Re}[]$代表复数的实部

$\text{Re}[\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q},\boldsymbol{k},m-n)$

然后我们用复数的指数形式，设：

$\begin{aligned} \boldsymbol{f}(\boldsymbol{q}, m) =&\, R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} \\ \boldsymbol{f}(\boldsymbol{k}, n) =&\, R_f (\boldsymbol{k}, n)e^{\text{i}\Theta_f(\boldsymbol{k}, n)} \\ \boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)e^{\text{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)} \\ \end{aligned}$

那么代入方程后就得到方程组：

$\begin{aligned} R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \\ \Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, n) =&\, \Theta_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \end{aligned}$

对于第一个方程，代入m=n得到：$R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, m) = R_g (\boldsymbol{q}, \boldsymbol{k}, 0) = R_f (\boldsymbol{q}, 0) R_f (\boldsymbol{k}, 0) = \Vert \boldsymbol{q}\Vert \Vert \boldsymbol{k}\Vert$最后一个等号源于初始条件$f(q,0)=q和f(k,0)=k$，所以现在我们可以很简单地设 $Rf(q,m)=∥q∥,Rf(k,m)=∥k∥$，即它不依赖于m。

至于第二个方程，同样代入m=n得到

$\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, m) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f (\boldsymbol{q}, 0) - \Theta_f (\boldsymbol{k}, 0) = \Theta (\boldsymbol{q}) - \Theta (\boldsymbol{k})$

这里的 $Θ(q),Θ(k)$是$q,k$本身的幅角，最后一个等号同样源于初始条件。根据上式得到 $Θf(q,m)−Θ(q)=Θf(k,m)−Θ(k)$，所以 $Θf(q,m)−Θ(q)$应该是一个只与m相关、跟q无关的函数，记为 $φ(m)$，即 $Θf(q,m)=Θ(q)+φ(m)$。接着代入 $n=m−1$，整理得到：

$\varphi(m) - \varphi(m-1) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta (\boldsymbol{k}) - \Theta (\boldsymbol{q})$

即$φ(m)$为等差数列，设右端为 $θ$，那么就解得 $φ(m)=mθ$。

# 3. 编码形式

综上，我们得到二维情况下用复数表示的RoPE：

$\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} = \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}$

根据复数乘法的几何意义，该变换实际上对应着向量的旋转，所以我们称之为“旋转式位置编码”，它还可以写成矩阵形式 （欧拉公式： $e^{i\Theta} = \cos\Theta + i\sin\Theta$）：

$\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}$

由于内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接，即：

$$ \begin{equation}\scriptsize{\underbrace{\begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ \end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}\end{equation} $$

也就是说，给位置为m的向量q乘上矩阵Rm、位置为n的向量k乘上矩阵Rn，用变换后的Q,K序列做Attention，那么Attention就自动包含相对位置信息了，因为成立恒等式：

$(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}$

值得指出的是，$Rm$是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

由于 $Rm$的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，推荐通过下述方式来实现RoPE：

$$ \begin{equation}\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} \end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} \end{pmatrix}\end{equation} $$

其中⊗是逐位对应相乘，即Numpy、Tensorflow等计算框架中的∗运算。从这个实现也可以看到，RoPE可以视为是乘性位置编码的变体。