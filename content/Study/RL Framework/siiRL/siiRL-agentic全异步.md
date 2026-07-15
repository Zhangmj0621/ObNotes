# 主pipeline

主pipeline文件在async_main.py文件中，核心类为class MainRunner，其中main函数中直接执行ray.get(runner.run())；

核心pipeline和常见异步框架并无差别，核心分为如下几步：

- 启动Task Coordinator，用来管理任务生命周期
- 分配GPU资源，为rollout和training分别分配对应的GPU资源
- 初始化data coordinator，每个节点分配一个distributed data buffer