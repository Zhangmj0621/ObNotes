## 1. 概述

本文使用slime中集成的retoot观察其工具调用能力

## 2. 实验流程

按照[slime quick start](https://www.notion.so/Slime-quick-start-269b4a3d835080d693a5d11cfdf77415?pvs=21)文档中，所述，准备好基础环境，本次实验以qwen3-4B-instruct为例

将hugging_face格式的qwen3-4B-instruct转换为megatron格式

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\
    ${MODEL_ARGS[@]} \\
    --hf-checkpoint /root/GLM-Z1-9B-0414 \\
    --save /root/GLM-Z1-9B-0414_torch_dist
```

执行对应的examples/retool路径下脚本即可

```bash
bash examples/retool/retool-qwen3-4b-rl.sh
```