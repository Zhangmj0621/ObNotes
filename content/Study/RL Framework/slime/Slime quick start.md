首先安装官方提供镜像

```bash
# Pull the latest image
docker pull zhuzilin/slime:latest

# start the docker
docker run -d -it --network=host --gpus all --privileged  --ipc=host --shm-size=16g   --ulimit memlock=-1 --ulimit stack=67108864  -v /model/:/workspace/infrawaves/  --name
 slime f01f4f265d5d  /bin/bash 
```

安装slime源码

```bash
git clone <https://github.com/THUDM/slime.git>
cd slime
pip install -e .
```

download模型与数据集、验证集

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com

# Download model weights (GLM-Z1-9B)
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \\
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \\
  --local-dir /root/aime-2024
```

转换hugging face权重为megatron格式

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\
    ${MODEL_ARGS[@]} \\
    --hf-checkpoint /root/GLM-Z1-9B-0414 \\
    --save /root/GLM-Z1-9B-0414_torch_dist
```

启动脚本

```bash
cd /root/slime
bash scripts/run-glm4-9B.sh
```

观察到会有如下的输出

![image.png](attachment:167f253b-97f1-414c-8d11-843b961ae4fd:image.png)

其中，核心的perf信息如下：

![image.png](attachment:d624f9a2-48db-4e6d-a37e-8cf5345ab8d9:image.png)

![image.png](attachment:0fe17281-1c78-47a5-a227-2220f52be9ca:image.png)