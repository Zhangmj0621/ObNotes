## Testbed

### 单机pd分离 1p1d

sglang 0.4.8.post1

```bash
# upgrade pip or uv
# install sglang
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.8.post1"

# install mooncake-transfer-engine
# you should ensure mooncake-transfer-engine version >= 0.3.4
uv pip install mooncake-transfer-engine
uv pip install mooncake-transfer-engine --upgrade

# prefill server
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode prefill --disaggregation-ib-device "roce10,roce11"

# decode server
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode decode --port 30001 --base-gpu-id 1 --disaggregation-ib-device "roce20,roce21"

# MiniLoadBalancer
python -m sglang.srt.disaggregation.mini_lb --prefill <http://127.0.0.1:30000> --decode <http://127.0.0.1:30001> --host 0.0.0.0 --port 8200

# bench
python3 -m sglang.bench_serving --backend sglang --dataset-name random  --dataset-path /root/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 8192 --random-output 1024 --num-prompts 200   --host 127.0.0.1 --port 8200    --model /model/Qwen2.5-7B-Instruct --pd-separated

# curl test
curl -s <http://localhost:8200/v1/chat/completions> \\
  -H "Content-Type: application/json" \\
  -d '{"model": "Qwen/Qwen2-1.5B-Instruct", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

### 2p2d using cache aware load balance

需要使用额外的package sglang_router，在sglang_router中，额外实现了不同的Schedule算法，例如power of two/cache aware(load balancing)/random

sglang_router: 0.1.5

```bash
# install sglang_router
pip install sglang_router -i   <https://pypi.mirrors.ustc.edu.cn/simple/>
```

```bash
# prefill 1 & prefill 2
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode prefill --base-gpu-id 6  --disaggregation-ib-device "mlx5_12,mlx5_13" --port 10000 --disaggregation-bootstrap-port 11000
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode prefill --base-gpu-id 4  --disaggregation-ib-device "mlx5_8,mlx5_9" --port 50000 --disaggregation-bootstrap-port 12000

# decode 1 & decode 2
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode decode --port 30002 --base-gpu-id 5 --disaggregation-ib-device "mlx5_10,mlx5_11" --port 40000
python -m sglang.launch_server --model-path /model/Qwen2.5-7B-Instruct/ --disaggregation-mode decode --port 30001 --base-gpu-id 7 --disaggregation-ib-device "mlx5_14,mlx5_15" --port 20000

# start router
# change policy for {random, cache_aware, power_of_two}
python -m sglang_router.launch_router   --prefill <http://127.0.0.1:10000> 11000 --prefill <http://127.0.0.1:50000> 12000  --decode  <http://127.0.0.1:20000> --decode <http://127.0.0.1:40000>  --pd-disaggregation --policy cache_aware

# bench serving
# change request-rate
python3 -m sglang.bench_serving --backend sglang --dataset-name random  --dataset-path /root/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 8192 --random-output 1024 --num-prompts 500   --host 127.0.0.1 --port 30000    --model /model/Qwen2.5-7B-Instruct --pd-separated --request-rate 5.0

```

### 16p8d

```bash
python -m sglang_router.launch_router --prefill <http://10.10.10.140:20000> 40000 --prefill <http://10.10.10.140:21000> 41000 --prefill <http://10.10.10.140:22000> 42000 --prefill <http://10.10.10.140:23000> 43000 --prefill <http://10.10.10.140:24000> 44000 --prefill <http://10.10.10.140:25000> 45000 --prefill <http://10.10.10.140:26000> 46000 --prefill <http://10.10.10.140:27000> 47000  --decode <http://10.10.10.150:20000> --decode <http://10.10.10.150:21000>  --decode <http://10.10.10.150:22000> --decode <http://10.10.10.150:23000> --decode <http://10.10.10.150:24000> --decode <http://10.10.10.150:25000> --decode <http://10.10.10.150:26000> --decode <http://10.10.10.150:27000>  --prefill <http://10.10.10.130:20000> 40000 --prefill <http://10.10.10.130:21000> 41000 --prefill <http://10.10.10.130:22000> 42000 --prefill <http://10.10.10.130:23000> 43000 --prefill <http://10.10.10.130:24000> 44000 --prefill <http://10.10.10.130:25000> 45000 --prefill <http://10.10.10.130:26000> 46000 --prefill <http://10.10.10.130:27000> 47000  --pd-disaggregation --policy cache_aware
```

## 非pd分离不同scheduling

random and cache_aware
![[Pasted image 20260408135109.png]]
![[Pasted image 20260408135121.png]]
Experiment

## scheduling

用rust编写，编译为对应的lib库

```bash
ls -l $(python -c "import site; print(site.getsitepackages()[0])")/sglang_router_rs*
-rwxr-xr-x 1 root root 21681513 Jul  7 12:07 /root/miniconda3/envs/sglang_zmj/lib/python3.12/site-packages/sglang_router_rs.cpython-312-x86_64-linux-gnu.so
```

代码位置在sglang/sgl-router/src