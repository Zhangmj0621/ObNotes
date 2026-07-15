### 1. 概述

本文针对siiRL中multi-turn的代码进行解读，主要详细针对其中agent和env交互的部分进行介绍，本文针对的pr为2025.8.28，其中只支持colocate模式的multi-turn，且env也只集成了非常基础的math_env，下面主要介绍几个主要的大改动

### 2. 环境定义

环境的基准类定义在文件siirl/workers/environment/base.py中，其中实现了抽象类BaseEnvironment，其中定义抽象接口reset(…)、step(…)与get_rewards(…)，提供给不同的env实现不同的接口，下面展示简单的math_env的接口实现

```python
from siirl.workers.environment.base import BaseEnvironment
from siirl.utils.reward_score.math import compute_score
from typing import Any, Dict, Optional, Tuple
class MathEnv(BaseEnvironment):
    def __init__(self):
        pass
    def reset(self) -> Any:
        pass
    def step(self, actions, ground_truth):
        actor_action = actions[-1]
        score = compute_score(actor_action, ground_truth)
        should_stop = False
        if score == 1.0:
            next_obs = [act + ". This answer is right." for act in actions]
            should_stop = True
        else:
            next_obs = [act + ". This answer is wrong." for act in actions]
        return next_obs, score, should_stop
```

其中，step函数实现直接根据actor给出的output，与ground_truth进行计算score，根据结果是否正确，添加对应response如”This answer is wrong.”

### 3. AgentProcess定义

AgentProcess用来管理单个agent的执行流程，其中封装了对于外界交互环境的管理
![[Pasted image 20260408135948.png]]
其中env_path指向具体定义的基础环境（如第二节中的MathEnv)，目前siiRL只支持单个环境，env_managers则是一个字典，将每个agent和他调用的环境进行绑定

同时，AgentProcess会指定具体的process_handle，一个可供参考的定义如下：

```python
from string import Template
def pre_process(tokenizer, prompt_id, obs, **kwargs):
	pre_chat_template = Template(kwargs.get("pre_chat_template", ""))
	prompt = tokenizer.decode(prompt_id)
	prompt = pre_chat_template.substitute(prompt = prompt)
	return tokenizer.encode(prompt)
	
def post_process(tokenizer, prompt_id, response_id, **kwargs):
	post_chat_template = kwargs.get("post_chat_template", None)
	post_chat_template_id = tokenizer.encode(post_chat_template)
	return prompt_id + post_chat_template_id + response_id
```

其中pre_process(…)将prompt从文本转变为具体的tensor，而post_process则返回prompt+特殊符号+response，在AgentProcess中，主要就是借助外部自定义的pre_process(…)与post_process(…)封装实现了接口`apply_pre_process(…)`与`apply_post_process(…)`

### 4. MultiAgentLoop实现

### 4.1. 调用时机

MultiAgentLoop的对象的定义在siirl/workers/dag_worker/mixins/initialization_mixin.py的`init_mode(…)`函数中，其中如下
![[Pasted image 20260408135959.png]]
在具体后续在siirl/workers/dag_worker/mixins/node_executors_mixin.py中的generate接口中，若self._multi_agent不为False，则调用multi_agent_loop中的`generate_sequence(..)`接口来生成rollout，其中推理引擎只支持sglang
![[Obsidian 2026-04-08 14.00.11.png]]
### 4.2. MultiAgentLoop定义

MultiAgentLoop的定义在siirl/workers/multi_agent/multiagent_generate.py中，先来看初始化

```python
class MultiAgentLoop(UtilitiesMixin):
    def __init__(self, dag, config: ActorRolloutRefArguments, node_workers:Dict, local_dag:TaskGraph, databuffer:List["ray.actor.ActorHandle"], placement_mode: str = 'colocate'):
        # dely import Dag after dagworker finish init
        from siirl.workers.dag_worker.dagworker import DAGWorker
        assert config.rollout.name == 'sglang', "MultiAgent only support sglang because vllm can't sleep in multi times"
        self.dag:DAGWorker = dag
        self.graph = local_dag
        self.placement_mode = placement_mode
        self.rollout_config = config.rollout
        self.internal_data_cache: Dict[str, Queue] = {}
        self.data_buffers = databuffer
        self.workers = node_workers
        self.max_model_len = int(self.rollout_config.max_model_len or self.rollout_config.prompt_length + self.rollout_config.response_length)
        self.max_model_len = min(self.max_model_len, self.rollout_config.prompt_length + self.rollout_config.response_length)
        self._parse_graph(local_dag)
        self.finish_generate = False 
        assert placement_mode == 'colocate' #in ['colocate', 'spread']
        if self.rollout_config.multi_turn.max_assistant_turns is None:
            self.rollout_config.multi_turn.max_assistant_turns = 1
```

目前，siirl强行要求采用colocated模式（不支持disaggregated架构），

下面详解`MultiAgentLoop::generate_sequence(…)`函数，generate_sequence(..)流程图如下：
![[Pasted image 20260408140024.png]]
其中`generate_colocate(…)`为核心函数，每个prompt掉用一次，一个batch内的所有prompt并行执行，generate_colocate的执行逻辑如下图，注意，其中的node_queue只包含rollout的node，如果不存在multiAgent，则len(node_queue)=1
![[Pasted image 20260408140033.png]]
其中，`colocate_task(…)`负责实际单个sample的output生成，其中包括与环境的交互，数据传递给下一个node等等。`colocate_task(…)`函数主要流程如下：

- 获取节点dp_size, dp_rank, tp_size, tp_rank等信息
- 获取下一个node对应的next_key, next_dp_size等信息(下一个agent）
- 检查agent_output.status，如果不为running，则保存rewards信息，调用`async_put_data(…)`将数据保存至下一个node对应的位置（data depency in workflow)
- 如果允许和环境交互，则调用async_get_envdata(…)获取和环境交互的response(obs)
- 预处理得到对应的templated_prompt，随后调用node_worker.rollout.generate(…)生成response，其中对于sglang，rollout即指sglangRollout类，具体的执行直接走`self.inference_engine.async_generate(…)`函数
- 将生成的结果进行后处理得到templated_prompt, response_mask等值
- 如果开启环境，则选择第一个可用的env，调用环境中的step(…)函数，对于math，则根据score在prompt后增加”your answer is wrong.”或”your answer is right.”，并且目前的siirl中只支持通过工具调用得到了正确答案后才返回
- 调用`async_put_data(…)`将next_obs存入(加上response)
- 如果为最后一个node，设置turn+1，如果超过了最大可用turn，则记为turn_FINISH
- 如果状态不为running，则调用`async_put_data(…)`将数据保存给下个node使用

`check_colocate_running(…)`则为检测单个sample是否被所有的node(Rollout)执行完成