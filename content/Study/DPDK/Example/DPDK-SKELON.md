# 注意事项

1. 注意，网卡配置要调成vmxnet3，在虚拟机文件.vmx中修改

```
ethernet0.virtualDev = "vmxnet3"
ethernet0.wakeOnPcktRcv = "TRUE"
```

1. 绑定网卡前需要激活模块vfio-pci

```
sudo modprobe vfio-pci

// 我也不知道为啥要修改权限
sudo /usr/bin/chmod a+x /dev/vfio
sudo /usr/bin/chmod 0666 /dev/vfio/*
```

1. 如何查看网卡目前配置

- 查看基本网卡信息

```
ifconfig -a
```

- 修改网卡绑定至vfio-pci上，能够用来dpdk使用

```
sudo python3 {dpdk_path}/usertools/dpdk-devbind.py --status	//查看目前绑定状态
sudo python3 {dpdk_path}/usertools/dpdk-devbind.py --unbind 03:00.0	//解绑卡

//解绑卡前需要提前将网卡关闭
sudo ifconfig ens33 down

sudo python3 {dpdk_path}/usertools/dpdk-bind.py --bind=vfio-pci ens33	//绑定
```

绑定成功后，你会看到类似如下的输出结果

![](https://cdn.nlark.com/yuque/0/2024/png/42492869/1722063620359-9d9167da-ad1d-44cb-a2af-aa3a60fa89c7.png)

1. 运行example代码

```
./<build_dir>/examples/dpdk-skeleton -l 1 -n 4
```

1. 特殊事项

- 需要开启大页内存
- 需要偶数个网卡绑定到dpdk上
- 需要启用IOMMU

丢个官方文档地址

[Getting Started Guide for Linux — Data Plane Development Kit 24.07.0-rc3 documentation (dpdk.org)](https://doc.dpdk.org/guides/linux_gsg/index.html)