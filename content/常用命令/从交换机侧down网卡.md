### Infiniband

以down 10.200.88.173 上的mlx5_gdr_0为例

在服务器上

iblinkinfo | grep "10-200-88-173" 或 iblinkinfo | grep "node073"

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=YjkwMmRiMDJkNmFkMDQxNjI0ZDAwN2ExZGM5MzZkZDhfaFpSME03Vm9qZkFJV1ByM1RmemJMR0hSN0FLVFR3TGRfVG9rZW46RlJGVGJZaXNKb0tLVEl4SW9UVWNZWnVlbnVoXzE3NDk4MjEwNjU6MTc0OTgyNDY2NV9WNA)

ibportstate -C mlx5_gdr_1 -P 1 157 13 disable

ibportstate -C mlx5_gdr_1 -P 1 157 13 enable

需要注意的是，在up端口时，指定的网卡需要是非与该端口相连的网卡，否则会由于找不到服务器端口而无法up，例如上面命令为down掉mlx5_gdr_0直接相连的交换机端口，但命令指定了mlx5_gdr_1(制定mlx5_gdr_2....7均可)

## Roce

### D1 & D2

首先登陆交换机，不同机子登陆方式不同

``` bash
ssh -o HostKeyAlgorithms=+ssh-rsa admin@172.171.2.201
enable

```

随后查看端口情况
``` bash
show lldp neighbors

```

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MGIyYjA0YjM3ZGM3MTQ5MGE2MTg1NjgyMjE3NjA4NThfWWsxaXhiOE41UG81NFU1eFAxalducHFycG5IbTVlZDZfVG9rZW46UVl3S2JSM1d2b3Y1SVJ4MzJGd2NXUmdUbjhlXzE3NDk4MjEwNjU6MTc0OTgyNDY2NV9WNA)
随后执行

```
configure
interface Fh0/6:1
shutdown // down port
no shutdown    // up port

```

### 北坡

从网页版进入交换机，执行命令

Show lldp table

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MjEyZjQ2OWIzNmMyMGY3ZGRmZGQyMTllMjBmYjBhZjJfS085YTFGSFhxSHZ6Rk9sNHppZ0Jta1NKaElGVWNOTFhfVG9rZW46QlcwTWJzdDB1b0RJQTN4THBKa2NENlFWbnhkXzE3NDk4MjEwNjU6MTc0OTgyNDY2NV9WNA)

找到对应想要down或up的port，执行命令

```
sudo config interface shutdown Ethernet248
sudo config interface startup Ethernet248

```