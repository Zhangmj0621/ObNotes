# 概述

本文简单介绍下verl中和gsm8k tool的交互方式

# 工具定义

工具定义config在 examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml中，具体gsm8k工具类在 verl/tools/gsm8k_tool.py中的Gsm8kTool

核心的函数create()和execute()定义如下：

```python
async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer

        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward

        return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}
```

整个gsm8k的工具调用非常简单，根据计算得到的answer和真实ground_truth进行比较，如果相等则reward为1，不然则为0，并输出对应的text