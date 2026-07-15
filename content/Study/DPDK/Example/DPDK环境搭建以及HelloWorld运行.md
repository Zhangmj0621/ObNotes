# 1. 下载DPDK

环境：

ubuntu 22.04

```
git clone <https://github.com/DPDK/dpdk.git>
```

# 2. 环境搭建

```
sudo apt-get install python3

sudo apt-get install python3-pip

sudo pip3 install meson

sudo pip3 install ninja
```

**注意**：

- meson 和 ninja是dpdk新版本的编译方式，且只能通过pip安装
- Python 3.6 或更高版本。
- Meson（版本 0.53.2+）和 ninja
- 大多数 Linux 发行版中的 meson 和 ninja-build 包
- 如果打包版本低于最低版本，可以从 Python 的“pip”仓库安装最新版本：pip3 install meson ninja pyelftools（版本 0.22+）
- 对于 Fedora 系统，可以使用 dnf install python-pyelftools 安装
- 对于 RHEL/CentOS 系统，可以使用 pip3 install pyelftools 安装
- 对于 Ubuntu/Debian，可以使用 apt install python3-pyelftools 安装
- 对于 Alpine Linux，可以使用 apk add py3-elftools 安装
- 用于处理 NUMA（非统一内存访问）的库。
- RHEL/Fedora 中的 numactl-devel；
- Debian/Ubuntu 中的 libnuma-dev；
- Alpine Linux 中的 numactl-dev

# 3. 编译dpdk

此处仅展示普通编译，交叉编译等等需求请查阅官网

```
//configure a DPDK build use
meson setup <options> build

//build and then install DPDK system-wide use
cd build
ninja
ninja install
ldconfig
```

注意：如果此过程中报错，查看是否是有包需要装，例如上一步所说的pyelftools等等

# 4. 设置大页内存（运行DPDK必须）

1. 编译GRUB配置文件
2. 打开终端
3. 编辑GRUB配置文件

```
sudo nano /etc/default/grub	//sudo is required
```

1. 修改GRUB配置文件
2. 在打开的文件中，找到GRUB_CMDLINE_LINUX_DEFAULT这一行，并在引号内添加你需要的hugepages参数。例如，如果你想预留1024个2MB的hugepages，修改如下：

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash hugepages=1024"
```

1. 如果你想预留4个1GB的hugepages，并且设置1GB为默认的hugepage大小，修改如下：

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash default_hugepagesz=1G hugepagesz=1G hugepages=4"
```

1. 更新GRUB

```
// 重新生成 grub.cfg
sudo grub-mkconfig -o /boot/grub/grub.cfg

// 更新 grub 配置
sudo update-grub

// 重启
reboot
```

1. 验证配置
2. 重启后，可通过如下命令验证hugepages是否以及正确预留

```
grep HugePages_Total /proc/meminfo
```

1. 手动挂载
2. 首先，创建一个目录作为Hugepages的挂载点

```
sudo mkdir -p /mnt/huge
```

1. 挂载hugepages：使用mount命令挂载hugepages。假设你像挂载1GB大小的hugepages：

```
sudo mount -t hugetlbfs nodev /mnt/huge -o pagesize=1G
```

1. 使挂载点在重启后永久生效

```
sudo nano /etc/fstab
```

在文件中添加如下行

```
nodev /mnt/huge hugetlbfs pagesize=1G 0 0
```

保存并关闭文件

1. 验证挂载

```
mount | grep hugetlbfs
```

如果能看到类似如下输出，则说明挂载成功

```
nodev on /mnt/huge type hugetlbfs (rw,relatime,pagesize=1G)
```

# 5. 加载Kernel driver

本文以VFIO为例

### 5.1 BIOS支持

在VMWARE设置中，将虚拟化INTEL VTX以及虚拟机IOMMU打开

验证是否成功

```
dmesg | grep -e DMAR -e IOMMU
```

输出非空白或者IOMMU NOT FOUND即为配置成功

### 5.2 内核支持

与hugepages类似，修改/etc/default/grub文件

```
在 GRUB_CMDLINE_LINUX_DEFAULT 里添加 iommu=pt intel_iommu=on
```

验证

```
cat /proc/cmdline | grep iommu=pt

cat /proc/cmdline | grep intel_iommu=on
```

### 5.3 加载kernel

```
sudo modprobe vfio-pci
sudo /usr/bin/chmod a+x /dev/vfio
sudo /usr/bin/chmod 0666 /dev/vfio/*
```

# 6. 绑定network ports

需要先关闭待绑定网卡

```
ifconfig ens33 down
```

随后进行绑定

```
sudo python3 {dpdk_path}/usertools/dpdk-devbind.py --bind=vfio-pci ens33
sudo python3 {dpdk_path}/usertools/dpdk-devbind.py --status	//验证
```

# 7. 运行hellowork example

### 7.1. 编译

进入dpdk build目录

```
cd dpdk/build
```

允许examples编译

```
meson configure -Dexamples=helloworld
```

编译

```
ninja
```

### 7.2. 运行

```
./<build_dir>/examples/dpdk-helloworld -l 0-3 -n 4
```

丢个官方文档地址

[Getting Started Guide for Linux — Data Plane Development Kit 24.07.0-rc3 documentation (dpdk.org)](https://doc.dpdk.org/guides/linux_gsg/index.html)