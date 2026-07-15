# MPI教程介绍

消息传递接口（Message Passing Interface, MPI）是一项对于分布式计算很有价值的技术。虽然MPI比大多数并行框架要更底层（比如Hadoop），但是学习MPI会为并行编程打下良好基础。

MPI教程的必要性

- 网上关于MPI的资料是过时的，或者是不全的
- 想要简单搭建一个可以运行的MPI的集群环境
- 关于MPI的书太贵了

## MPI历史简介

90年代之前，对于不同的计算架构写并发程序是件困难和冗长的事情，很多软件库可以帮忙编写并发程序，然而并不存在一个大家都接受的标准。

最广为接受的模型是消息传递模型，消息传递模型是指程序通过在进程间传递消息来完成某些任务，例如，主进程可以通过对从进程发送一个描述工作的消息来把这个工作分配给它。另一个例子是一个并发的排序程序可以在当前进程中对当前进程可见的（我们称作本地的，locally）数据进行排序，然后把排好序的数据发送到邻居进程上面来进行合并的操作。几乎所有的并行程序可以使用消息传递模型来描述。

很多软件库用到了这个消息传递模型，但在定义上有微小差异，于是1992年定义了一个消息传递接口的标准-MPI。

到1994年，一个完整的接口标准定义好了（MPI-1)，MPI只是一个借口定义，需要程序员根据不同的架构来实现这个接口，幸运的是一年后一个完整的MPI实现就已经出现了，在第一个实现之后，MPI就被大量的使用在消息传递应用程序中。

## MPI对于消息传递模型的设计

相关概念介绍

通讯器（communicator）定义了一组能够互相发消息的进程。在这组进程中，每个进程会被分配一个序号，称作秩（rank），进程间显性地通过指定秩来进行通信。

通信的基础建立在不同进程间发送和接收操作。一个进程可以通过指定另一个进程的秩以及一个独一无二的消息标签（tag）来发送消息给另一个进程。接受者可以发送一个接收特定标签标记的消息的请求（或者也可以完全不管标签，接收任何消息），然后依次处理接收到的数据。类似这样的涉及一个发送者以及一个接受者的通信被称作点对点（point-to-point）通信。

当然在很多情况下，某个进程可能需要跟所有其他进程通信。比如主进程想发一个广播给所有的从进程。在这种情况下，手动去写一个个进程点对点的信息传递就显得很笨拙。而且事实上这样会导致网络利用率低下。MPI 有专门的接口来帮我们处理这类所有进程间的**集体性（collective）通信**。

把点对点通信和集体性通信这两个机制合在一起已经可以创造十分复杂的并发程序了。事实上，这两个功能已经强大到我现在不需要再介绍任何 MPI 高级的特性了。

# 在单机上下载MPICH2

MPI有众多实现，本文仅使用MPICH（最受欢迎的MPI），此外，例如OpenMPI之类的MPI也都可以使用。

## 下载MPICH

最新版本地址https://www.mpich.org/

下载源代码，解压缩文件夹，然后切换到MPICH目录

```
>>> tar -xzf mpich3-3.2.tar.gz
>>> cd mpich3-3.2

```

完成之后，执行./configure来配置安装，若需要将MPICH安装到本地，请使用`./configure --prefix=/installation/directory/path`，如果您没有 Fortran 编译器，则可以通过使用 `./configure --disable-fortran` 来避免构建 MPI Fortran 库。输入 `./configure --help` 以获取有关可能的配置参数的更多信息。

```
>>> ./configure --disable-fortran
Configuring MPICH version 3.3.2
Running on system: Linux localhost.localdomain 5.8.18-100.fc31.x86_64 #1 SMP Mon Nov 2 20:32:55 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
checking build system type... x86_64-unknown-linux-gnu

```

配置完成后，应显示 “Configuration completed.” 一旦完成，就该使用make; sudo make install命令来构建和安装 MPICH2 了。

```
>>> make; sudo make install
make
make  all-recursive

```

如果构建成功，则应该可以输入mpiexec --version并看到以下类似的内容。

```
>>> mpiexec --version
HYDRA build details:
    Version:                         3.3.2
    Release Date:                    Tue Nov 12 21:23:16 CST 2019
    CC:                              gcc
    CXX:                             g++
    F77:                             gfortran
    F90:                             gfortran

```

# MPI hello world

本节展示一个基础的MPI hello world示例，本节代码运行在MPICH2，代码在https://github.com/mpitutorial/mpitutorial/tree/gh-pages/tutorials/mpi-hello-world/code

## Hello World代码

```
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 初始化 MPI 环境
    MPI_Init(NULL, NULL);

    // 通过调用以下方法来得到所有可以工作的进程数量
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 得到当前进程的秩
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 得到当前进程的名字
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // 打印一条带有当前进程名字，秩以及
    // 整个 communicator 的大小的 hello world 消息。
    printf("Hello world from processor %s, rank %d out of %d processors\\n",
           processor_name, world_rank, world_size);

    // 释放 MPI 的一些资源
    MPI_Finalize();
}

```

1. 引入`<mpi.h>`头文件
2. 利用`MPI_INIT`初始化，所有的MPI全局变量或者内部变量都会被创建，举例来说，一个通讯器 communicator 会根据所有可用的进程被创建出来（进程是我们通过 mpi 运行时的参数指定的），然后每个进程会被分配独一无二的秩 rank。当前来说，`MPI_Init` 接受的两个参数是没有用处的，不过参数的位置保留着，可能以后的实现会需要用到。
3. `MPI_Comm_size`返回communicator的大小，也就是communicator中可用的进程数量，在我们的例子中，`MPI_COMM_WORLD`（这个 communicator 是 MPI 帮我们生成的）这个变量包含了当前 MPI 任务中所有的进程，因此在我们的代码里的这个调用会返回所有的可用的进程数目。
4. `MPI_Comm_rank` 这个函数会返回 communicator 中当前进程的 rank。 communicator 中每个进程会以此得到一个从0开始递增的数字作为 rank 值。rank 值主要是用来指定发送或者接受信息时对应的进程。
5. `MPI_Get_processor_name` 会得到当前进程实际跑的时候所在的处理器名字。
6. `MPI_Finalize` 是用来清理 MPI 环境的。这个调用之后就没有 MPI 函数可以被调用了。

## 运行MPI hello world程序

Makefile

```
EXECS=mpi_hello_world
MPICC?=mpicc

all: ${EXECS}

mpi_hello_world: mpi_hello_world.c
    ${MPICC} -o mpi_hello_world mpi_hello_world.c

clean:
    rm ${EXECS}

```

若MPICH2安装在本地，则手动设置一下MPICC这个环境变量，指向本地mpicc二进程程序，mpicc实际上只对gcc做了一层封装

```
>>> export MPICC=/home/kendall/bin/mpicc
>>> make
/home/kendall/bin/mpicc -o mpi_hello_world mpi_hello_world.c

```

当程序编译好之后，就可以被执行了，若是需要在好几个节点的集群上运行MPI程序的话，则需要配置一个host文件（不是/etc/hosts），若在单机上运行则可以忽略。

需要配置的host文件会包含想要运行的所有节点名称，为了运行方便，需要确认所有节点之间能通过ssh通信，并且需要根据

[http://www.eng.cam.ac.uk/help/jpmg/ssh/authorized_keys_howto.html配置不需要密码的ssh访问，可能的host文件如下](http://www.eng.cam.ac.uk/help/jpmg/ssh/authorized_keys_howto.html%E9%85%8D%E7%BD%AE%E4%B8%8D%E9%9C%80%E8%A6%81%E5%AF%86%E7%A0%81%E7%9A%84ssh%E8%AE%BF%E9%97%AE%EF%BC%8C%E5%8F%AF%E8%83%BD%E7%9A%84host%E6%96%87%E4%BB%B6%E5%A6%82%E4%B8%8B)

```
>>> cat host_file
cetus1
cetus2
cetus3
cetus4

```

为了使用脚本来运行程序，需要设置一个叫做MPI_HOSTS的环境变量，把他纸箱host文件所在的位置，脚本会自动将host文件的配置项加到MPI启动命令里，若单机运行则不需要，此外若MPI安装到本地目录，则需要指定MPIRUN这个环境变量。

执行脚本在tutorials目录下，可以在mpitutorial这个文件夹的根目录下执行以下命令

```
>>> export MPIRUN=/home/kendall/bin/mpirun
>>> export MPI_HOSTS=host_file
>>> cd tutorials
>>> ./run.py mpi_hello_world
/home/kendall/bin/mpirun -n 4 -f host_file ./mpi_hello_world
Hello world from processor cetus2, rank 1 out of 4 processors
Hello world from processor cetus1, rank 0 out of 4 processors
Hello world from processor cetus4, rank 3 out of 4 processors
Hello world from processor cetus3, rank 2 out of 4 processors

```

MPI程序运行在了提供的所有节点上面，每个进程被分配了一个单独的rank，，在输出结果中可以发现，进程之间打印顺序是任意的，代码里并不涉及同步的操作。

从打印log可知，脚本创建了四个进程，对于每台机器都执行MPI程序

若节点是双核机器，如何让MPI先在每个节点上的每个核生成进程，再去其他的机器，方案为修改host文件，在每个节点名字后添加冒号和每个处理器有的核数即可，例如，在host文件里指定每个节点有2个核

```
>>> cat host_file
cetus1:2
cetus2:2
cetus3:2
cetus4:2

```

再次运行脚本

```
>>> ./run.py mpi_hello_world
/home/kendall/bin/mpirun -n 4 -f host_file ./mpi_hello_world
Hello world from processor cetus1, rank 0 out of 4 processors
Hello world from processor cetus2, rank 2 out of 4 processors
Hello world from processor cetus2, rank 3 out of 4 processors
Hello world from processor cetus1, rank 1 out of 4 processors

```

# 在Amazon EC2上构建和运行自己的集群

如何设置私有的虚拟MPI集群

## 利用Amazono EC2开始

集群将使用Amazon的Elastic Compute Cloud（EC2），允许从Amazon的基础设置租用虚拟机，开始使用Amazon EC2，可以转到Amazon Web Services（[https://aws.amazon.com/cn/）进行注册。](https://aws.amazon.com/cn/%EF%BC%89%E8%BF%9B%E8%A1%8C%E6%B3%A8%E5%86%8C%E3%80%82)

在注册AWS后，需要阅读EC2入门指南，来熟悉如何启动、访问和终止不同的机器实例

## 安装StarCluster

创建虚拟机群的工具为MIT的StarCluster工具包，StarCluster是一套工具，可自动执行在EC2上构建和访问集群的过程。除了启动集群外，StarCluster还会安装OpenMPI和其他用于编写并行应用程序的软件。

需要使用python在本地计算机安装StarCluster工具包，以linux环境为例

```
$ sudo easy_install StarCluster

```

## 配置StarCluster

安装StarCluster后，输入

```
$ starcluster help

```

由于StarCluster尚未配置，因此将打印出类似如下内容

```
StarCluster - (<http://web.mit.edu/starcluster>) (v. 0.93.3)
Software Tools for Academics and Researchers (STAR)
Please submit bug reports to starcluster@mit.edu
!!! ERROR - config file /Users/wesleykendall/.starcluster/config does not exist
Options:
--------
[1] Show the StarCluster config template
[2] Write config template to /Users/wesleykendall/.starcluster/config
[q] Quit
Please enter your selection:

```

输入数字2，StarCluster将在你的目录下生成一个默认配置文件，~/.starcluster/config

生成默认配置后，从AWS账号可获取AWS访问密钥、秘密访问密钥和12位用户ID。

打开配置文件，找到以下行【aws info】，将所有AWS信息输入到适合的字段中

```
[aws info]
AWS_ACCESS_KEY_ID = # Your Access Key ID here
AWS_SECRET_ACCESS_KEY = # Your Secret Access Key here
AWS_USER_ID = # Your 12-digit AWS Account ID here (no hyphens)

```

输入次信息后，保存配置文件，然后创建一个公钥/私钥对，将该密钥对上传到Amazon，并在登录集群时用于验证你的机器，调用以下命令使用StarCluster生成公钥/私钥对

```
$ starcluster createkey mykey -o ~/.ssh/mykey.rsa

```

这将在计算机上创建一个“mykey"密钥~/.ssh/mykey.rsa，并在您的AWS账户上创建一个密钥，如果正确输入了Amazon凭证，您应该可以看到类似以下内容输出：

```
>>> Successfully created keypair: mykey
>>> fingerprint: ...
>>> contents:
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----

```

接下来，再次打开配置文件以验证是否有条目【key mykey】。若默认配置文件没有此条目，请将以下内容添加到您的配置中：

```
[key mykey]
KEY_LOCATION = ~/.ssh/mykey.rsa

```

下一步（最后一步）配置涉及配置集群参数。默认配置以及在【cluster smallcluster】配置行中定义了一个名为”smallcluster“的集群。它应该具有以下默认参数：

```
[cluster smallcluster]
KEYNAME = mykey
CLUSTER_SIZE = 2
CLUSTER_USER = sgeadmin
CLUSTER_SHELL = bash
NODE_IMAGE_ID = ami-899d49e0
NODE_INSTANCE_TYPE = m1.small

```

字段概述如下：如果要启动具有两个以上节点的集群，请更改选项`CLUSTER_SIZE`。若您定义了不同的键，请确保添加适当的键作为字段`KEYNAME`。运行集群时，它将自动为`CLUSTER_USER`您生成一个用户名，该用户名具有通过网络文件系统（NFS)安装的主目录。是`NODE_IMAGE_ID`与所有集群软件一起打包的虚拟机映像。最后一个参数`NODE_INSTANCE_TYPE`确定每个节点的大小。

### 为StarCluster启用mpich2插件

最后，在启动StarCluster之前，请请`mpich2`按照[以下步骤](http://star.mit.edu/cluster/docs/0.93.3/plugins/mpich2.html)启用 Starcluster 插件。

## 启动、访问和停止集群

配置StarCluster后，键入以下命令启动名为”mpicluster"的集群，默认配置使用"smallcluster"作为默认集群类型：

```
starcluster start mpicluster

```

启动集群的过程可能需要一些时间，具体取决于您的配置，命令完成后，StarCluster将打印出可用于访问、停止和重新启动集群的命令

输入以下命令通过 SSH 进入集群的管理器节点：

```
starcluster ssh manager mpicluster

```

登录集群后，您当前的工作目录将是/root。切换到/home/ubuntu或/home/sgeadmin区域以编译代码。这些目录安装在网络文件系统上，集群中的所有节点都可以查看。

当您位于已挂载的主目录中时，请继续从其 GitHub 存储库中查看 MPI 教程代码。本网站上的每节课都使用该代码：

```
git clone git://github.com/mpitutorial/mpitutorial.git

```

当您感觉可以顺利访问集群后，请注销并停止集群：

```
starcluster stop mpicluster

```

可以通过输入以下命令重新启动已停止的集群：

```
starcluster start -x mpicluster

```

要完全终止集群，请输入：

```
starcluster terminate mpicluster

```

“已停止”和“已终止”集群之间的区别在于，已停止的集群仍作为映像驻留在 Amazon 的 Elastic Block Store (EBS) 上。如果您在可预见的未来不会再次启动集群，建议继续终止集群，因为 Amazon EBS 付款费率适用于存储的实例。与往常一样，在开始之前先了解Amazon EC2 定价。

# 在LAN中运行MPI集群

之前研究了如何在单机上运行MPI程序来并行处理代码，充分利用CPU中的多个核心，现在稍微扩大下范围，将同样的操作从多台计算机扩展到局域网中相互连接的节点网络，本文为简单起见，我们现在只考虑两台计算机

## 配置hosts文件

需要在计算机间进行通信，但不想输入IP地址，相反，可以为网络中要通信的各个节点命名，hosts设备操作系统会使用该文件将文件名映射到IP地址

```
$ cat /etc/hosts
127.0.0.1       localhost
172.50.88.34    worker

```

这worker是您要用来进行计算的机器

## 创建新用户

建议创建一个新用户以简化配置，创建一个新用户mpiuser，在所有机器中创建具有相同用户名的新用户账号，以简化操作

```
$ sudo adduser mpiuser

```

请不要使用useradd来创建新用户，因为这样不会为新用户创建单独的主页

## 设置ssh

机器将通过SSH在网络上进行通信并通过NFS共享数据，稍后会讨论这个问题

```
$ sudo apt­-get install openssh-server

```

随后使用新创建账号登录

```
$ su - mpiuser

```

由于ssh服务器已经安装，您必须能够通过 登录到其他机器ssh username@hostname，此时系统将提示您输入 的密码username。为了更轻松地登录，我们生成密钥并将其复制到其他机器的 列表中authorized_keys。

```
ssh-keygen -t dsa
$ ssh-copy-id worker #ip-address may also be used

```

对每台工作机器和您自己的用户（localhost）执行上述步骤。

这将为openssh-server您设置与工作机器的安全通信。ssh所有机器一次，以便它们被添加到您的列表中known_hosts。这是一个非常简单但必不可少的步骤，如果没有密码，ssh就会有麻烦。

现在，要启用无密码 ssh，

```
$ eval `ssh-agent`$ ssh-add ~/.ssh/id_dsa

```

现在，假设您已正确将密钥添加到其他机器，则您必须能够在无需任何密码提示的情况下登录到其他机器。

```
$ ssh worker

```

## 设置NFS

通过管理器中的NFS共享一个目录，工作其会挂载该目录来交换数据

### NFS服务器

安装所需软件包

```
 sudo apt-get install nfs-kernel-server

```

现在，（假设您仍处于登录状态），让我们创建一个将在网络上共享`mpiuser`的同名文件夹。`cloud`

```
$ mkdir cloud

```

要导出`cloud`目录，请在`/etc/exports`

```
$  cat /etc/exports
/home/mpiuser/cloud *(rw,sync,no_root_squash,no_subtree_check)

```

在这里，`*`您可以具体给出要共享此文件夹的 IP 地址。但这只会让我们的工作更轻松。

- **rw ：**这是启用读写选项。ro为只读。
- **sync**：仅当提交更改后，才会将更改应用于共享目录。
- **no_subtree_check**：此选项可防止子树检查。当共享目录是较大文件系统的子目录时，nfs 将扫描其上层的每个目录，以验证其权限和详细信息。禁用子树检查可能会提高 NFS 的可靠性，但会降低安全性。
- **no_root_squash**：这允许 root 帐户连接到该文件夹。

输入完成后，运行以下命令。

```
$ exportfs -a

```

每次对 进行更改时，运行上述命令`/etc/exports`。

如果需要，重新启动`nfs`服务器

```
$ sudo service nfs-kernel-server restart

```

### NFS worker

安装所需的软件包

```
$ sudo apt-get install nfs-common

```

在 worker 的机器上创建一个同名的目录`cloud`

```
$ mkdir cloud

```

现在，挂载共享目录，如下所示

```
$ sudo mount -t nfs manager:/home/mpiuser/cloud ~/cloud

```

要检查已挂载的目录，

```
$ df -h
Filesystem                          Size  Used Avail Use% Mounted on
manager:/home/mpiuser/cloud  49G   15G   32G  32% /home/mpiuser/cloud

```

为了使挂载永久生效，这样您不必在每次重新启动系统时手动挂载共享目录，您可以在文件系统表中创建一个条目 - 即`/etc/fstab`如下文件：

```
$ cat /etc/fstab
#MPI CLUSTER SETUP
manager:/home/mpiuser/cloud /home/mpiuser/cloud nfs

```

## 运行MPI程序

仅使用 MPICH2 安装包附带的一个示例程序`mpich2/examples/cpi`。我们将采用此可执行文件并尝试并行运行它。

或者如果您想编译自己的代码，假设其名称为`mpi_sample.c`，您将按照下面给出的方式进行编译，以生成可执行文件`mpi_sample`。

```
$ mpicc -o mpi_sample mpi_sample.c

```

首先将您的可执行文件复制到共享目录中`cloud`，或者更好的是，在 NFS 共享目录中编译您的代码。

```
$ cd cloud/
$ pwd
/home/mpiuser/cloud

```

要仅在您的机器上运行它，您可以这样做

```
$ mpirun -np 2 ./cpi No. of processes = 2

```

现在，要在集群中运行它，

```
$ mpirun -np 5 -hosts worker,localhost ./cpi
#hostnames can also be substituted with ip addresses.

```

或者在主机文件中指定相同的内容，然后

```
$ mpirun -np 5 --hostfile mpi_file ./cpi

```

**这应该会在您的管理器**所连接的所有机器上启动您的程序。