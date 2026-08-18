Mooncake编译步骤如下，以GB300为例
### 有网络跳板机机
```py title=""
# ① yaml-cpp 的 arm64 deb —— 在 arm64 容器里下，保证架构和版本对
docker run --rm --platform linux/amd64 -v $PWD:/out ubuntu:24.04 bash -c '
  set -e
  rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports noble main universe" \
    > /etc/apt/sources.list.d/arm64.list
  dpkg --add-architecture arm64
  apt-get update
  cd /out
  apt-get download libyaml-cpp-dev:arm64 libyaml-cpp0.8:arm64
  ls -l *.deb
'

# ② 两个 submodule 源码（纯文本，架构无关）
git clone https://github.com/alibaba/yalantinglibs.git
cd yalantinglibs && git checkout 7801bc9ad9021781f15217552214e325a1cf7373 && cd ..

git clone https://github.com/pybind/pybind11.git
cd pybind11 && git checkout 58c382a8e3d7081364d2f5c62e7f429f0412743b && cd ..

# ③ 打包（去掉 .git 减体积）
tar czf mooncake-deps.tar.gz \
  --exclude='.git' \
  yalantinglibs pybind11 *.deb
```
### GB300
```py title=""
cd /mnt/nfs/zhangmj        # 你 scp 的目标路径
tar xzf mooncake-deps.tar.gz

# ① 内容完整性 —— ylt 的关键头文件必须在
ls yalantinglibs/include/ylt/ | head
ls yalantinglibs/cmake/ 2>/dev/null
test -f yalantinglibs/CMakeLists.txt && echo "ylt CMakeLists OK"

# ② pybind11
test -f pybind11/CMakeLists.txt && echo "pybind11 CMakeLists OK"
ls pybind11/include/pybind11/pybind11.h

# ③ deb 架构
ls -l *.deb

.git 被剥掉了，所以在 GB300 上没法再用 git rev-parse 复查 commit —— 你已经在联网机器上确认过（HEAD is now at 7801bc9 和 58c382a8），那就够了。

然后走原来的四段



dpkg -i libyaml-cpp0.8_0.8.0+dfsg-6build1_arm64.deb \
        libyaml-cpp-dev_0.8.0+dfsg-6build1_arm64.deb

ls /usr/include/yaml-cpp/yaml.h
find /usr -name "yaml-cpp*onfig.cmake" 2>/dev/null

第 2 段：放 submodule + 装 ylt

注意路径 —— 你的 Mooncake 在 /workspace/infrawaves/zhangmnfs/zhangmj：

cd /workspace/infrawaves/zhangmj/Mooncake
rm -rf extern/yalantinglibs extern/pybind11
cp -r /mnt/nfs/zhangmj/yalantinglibs extern/yalantinglibs
cp -r /mnt/nfs/zhangmj/pybind11      extern/pybind11

cd extern/yalantinglibs
mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --install .

find /usr/local -iname "yalantinglibs*onfig.cmake" 2>/dev

第 3 段：配置（跑完停下）

cd /workspace/infrawaves/zhangmj/Mooncake
rm -rf build && mkdir build && cd build

cmake -G Ninja .. \
  -DUSE_CUDA=ON -DUSE_MNNVL=ON -DUSE_HTTP=ON \
  -DWITH_EP=OFF -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_BENCHMARK=OFF \
  -DCMAKE_BUILD_TYPE=Release

grep USE_MNNVL CMakeCache.txt     # 必须 USE_MNNVL:BOOL=ON

第 4 段：编译 + 替换

ninja engine
ldd mooncake-integration/engine.*.so | grep "not found"

PKG=/usr/local/lib/python3.12/dist-packages/mooncake
cp $PKG/engine.so ~/engine.so.bak
cp mooncake-integration/engine.*.so $PKG/engine.so

python -c "from mooncake.engine import TransferEngine as e_remote_mappings'))"
```
