actor：训练的模型

reward：绝对位置打分，response对于是否符合要求的打分，例如math上要求输入(10-20)的一个数字

reference：根据初始model的对比，约束模型不要偏离初始状态太远，根据reward可以计算KL散度

critic：相对位置，response对于所有可能的responce的打分

其中 $\pi _{\theta_{old}}$ 代表旧策略下生成动作的概率， $\pi_\theta$代表新策略 ，每一个step后权重更新，对于grpo，则是8个response共享一个新策略
![[Pasted image 20260408130653.png]]
$A_t$代表相对打分，包含reward model给出的绝对打分，以及critic model给出的罚分
![[Pasted image 20260408130708.png]]
clip代表截断，reference model即是下公式体现，限制新旧策略差异，**旧策略一般是指SFT后model，而不是上一个step的model**
![[Pasted image 20260408130719.png]]