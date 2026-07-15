# Perliminary
一个TP场景下的计算+通信核心分为如下几个部分，可以看到，先是qkv的projection，随后做具体的attention计算，然后会做o_projection，随后是AllReduce+RMSNorm+Residual Addition，随后进入MLP阶段，分别是MLP-1, Activation, MLP-2, AllReduce+RMSNorm+Residual Addition；
![[Pasted image 20260715202551.png]]
# Paper
## FLUX: FAST SOFTWARE-BASED COMMUNICATION OVERLAP ON  GPUS THROUGH KERNEL FUSION

## TOKENWEAVE: EFFICIENT COMPUTE-COMMUNICATION OVERLAP FOR  DISTRIBUTED LLM INFERENCE