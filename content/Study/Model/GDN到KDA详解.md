# 概述
本文概述linear attention中核心组件GDN和KDA的具体组成；Linear attention的motivation很显然易见，对于传统transformer模型，随着seq_len的不断scale，kvcache压力按照o(n)增大， 而计算attention的复杂度则完全超线性增长，因此，期望有新计算范式的出现；
# GDN
