# ~~SOP脚本~~ 未完善

```
# 裸金属环境  传参：要拉取的ngc镜像版本号
apt-get install libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit nvidia-container-toolkit-base nvidia-container-runtime nvidia-docker2 -y

docker pull nvcr.io/nvidia/pytorch:$1-py3

#国内下载方式
<https://github.com/DaoCloud/public-image-mirror>
docker pull m.daocloud.io/nvcr.io/nvidia/pytorch:24.07-py3
S74Y8UR&;BihD8ENbM4zF9ip/?#Q2+@I

```

```
# docker内操作
# 创建文件夹
mkdir infrawaves

```

# 配置环境 拉取镜像 并创建基础docker

```
# 基础依赖
apt-get install libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit nvidia-container-toolkit-base nvidia-container-runtime nvidia-docker2 -y

# 拉取需要版本的NGC镜像
docker pull nvcr.io/nvidia/pytorch:$1-py3

docker login --username=aliyun8978008701crpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com

docker pullcrpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/infrawaves-aicloud/ngc:megatron-kwai-atc24-3

docker pullcrpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/infrawaves-aicloud/24.10-singleport-nvshmem-3-2-5

docker pullcrpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/infrawaves-aicloud/ngc:24.02

docker run -d -it --network=host --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864     --name  infra_test nvcr.io/nvidia/pytorch:24.02-py3

docker run -d -it --network=host --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864   -v ./infra:/workspace/  --name  infra_test nvcr.io/nvidia/pytorch:24.02-py3

docker run -d -it --network=host --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -v /mnt/nfs/megatrace:/workspace   --name  megatrace_test  imegatrace:24.02

# docker run -d -it --network=host --runtime=nvidia   --gpus all    --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -v ./infra:/workspace   --name  j_test   crpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/infrawaves-aicloud/ngc:24.02  （5090 可能是新版  nvidia-container-toolkit 传参方式）

```

# 进到docker内做二次配置

## SSH

```
apt-get update
apt-get install openssh-server -y
apt-get install net-tools -y

vim /etc/ssh/sshd_config
Port 3217

vim /etc/ssh/ssh_config
StrictHostKeyChecking no
Port 3217

vim ~/.bashrc
/etc/init.d/ssh restart

#/bin/bash
#-c
#/etc/init.d/ssh restart && sleep inf

```

/etc/init.d/ssh restart

4、ssh-keygen 生成秘钥 并制作免密

把自己的公钥cat ~/.ssh/id_rsa.pub 放到

~/.ssh/authorized_keys

自己登录自己一次

## TE依赖编译 (不启用tp overlap可以不做)

```
export NVTE_FRAMEWORK=pytorchexport NVTE_WITH_USERBUFFERS=1export MPI_HOME=/usr/local/mpi
git clone --branch stable --recursive <https://gitee.com/jiaoff_gitee/TransformerEngine.git>
cd TransformerEngine
pip install .# Build and install

```

## 升级proc（可选 NGC 24.07版本多机需要 ）

把安装包传到docker里面

暂时无法在飞书文档外展示此内容

暂时无法在飞书文档外展示此内容

pip install protobuf-3.19.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

unzip protoc-3.19.0-linux-x86_64.zip -d /usr/local

# 打包和导入

```
# 提交
docker commit your_container_id infrawaves_test:24.07
# 打包
docker save -o infrawaves_test_24.07.tar infrawaves_test:24.07
#拷贝到共享存储  在所有GPU机器上加载
docker load -i infrawaves_test_24.07.tar

docker run -d -it --network=host --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/nfs/megatrace/:/workspace    --name megatrace ngc:24.03

nvcr.io/nvidia/mlperf/mlperf-inference                                                       mlpinf-v4.1-cuda12.4-pytorch24.04-ubuntu22.04-x86_64-release   4589a961919a   9 months ago    71.4GB

```