清除脚本

```bash
#!/bin/bash

USER="root"

# 节点列表
NODES=(
    172.27.0.72
    172.27.0.73
    172.27.0.74
    172.27.0.75
    172.27.0.77
    172.27.0.78
    172.27.0.79
    172.27.0.80
    172.27.0.81
    172.27.0.82
    172.27.0.83
    172.27.0.84
    172.27.0.85
    172.27.0.86
    172.27.0.87
    172.27.0.88
)

CONTAINER_NAME="aio"

for TARGET in "${NODES[@]}"; do
    echo "====== 连接 ${TARGET} ======"
    ssh -o StrictHostKeyChecking=no ${USER}@${TARGET} "
        if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}\\$'; then
            echo '进入 ${TARGET} 容器 ${CONTAINER_NAME} 并杀掉 ray 和 python 进程'
            docker exec ${CONTAINER_NAME} bash -c '
                pkill -f ray || ray stop --force;
                pkill -f python
            '
            echo '已成功杀掉 ray 和 python 进程'
        else
            echo '[${TARGET}] 容器 ${CONTAINER_NAME} 不存在，跳过该节点！'
        fi
    " &
done

wait
echo "全部节点执行完毕"
```

启动ray

```bash
#!/bin/bash

USER="root"

# 节点列表
NODES=(
    172.27.0.73
    172.27.0.74
    172.27.0.75
    172.27.0.77
    172.27.0.78
    172.27.0.79
    172.27.0.80
    172.27.0.81
    172.27.0.82
    172.27.0.83
    172.27.0.84
    172.27.0.85
    172.27.0.86
    172.27.0.87
    172.27.0.88
)

CONTAINER_NAME="aio"

for TARGET in "${NODES[@]}"; do
    echo "====== 连接 ${TARGET} ======"
    ssh -o StrictHostKeyChecking=no ${USER}@${TARGET} "
        if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}\\$'; then
            echo '进入 ${TARGET} 容器 ${CONTAINER_NAME} 并启动ray服务'
            docker exec ${CONTAINER_NAME} bash -c '
                ray start --address='172.27.0.72:7379' --num-gpus=8
            '
            echo '已成功启动ray服务'
        else
            echo '[${TARGET}] 容器 ${CONTAINER_NAME} 不存在，跳过该节点！'
        fi
    " &
done

wait
echo "全部节点执行完毕"
```