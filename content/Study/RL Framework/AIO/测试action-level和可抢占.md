```py title=""
 公共 wiring (两种场景都要)                                                                                                                     
                                                                                                                                                 
  config.rollout 里：                                                                                                                            
  - executor_module: "naive" —— 保留默认，走 action-level 路径会自动选到 AgenticExecutor (rollout_worker.py:126-129)                             
  - enable_action_level_scheduling: true                                                                                                         
  - flow_function: "agent" —— 才会走 build_agentflow(...) 注册表                                                                                 
  - flow_config: <你的 yaml>                                                                                                                     
                                                                                                                                                 
  flow_config yaml 里:                                                                                                                           
  - flow name: action_level_swe (对应 siirl/execution/rollout/agentflow/__init__.py:10 注册表)                                                   
  - agent name: action_level_sii_sweagent (对应 siirl/execution/rollout/agentflow/swe/__init__.py:169)                                           
                                                                                                                                                 
  场景 A：只开 action-level，不抢占                                                                                                              
                                                                                                                                                 
  在公共 wiring 基础上，保持：                                                                                                                   
  - enable_priority_based_scheduling: false (默认)                                                                                               
  - priority_scheduling_policy: 无所谓，不会被读                                                                                                 
                                                
  运行时表现：AgenticExecutor._enable_preemption == False，SchedState / 优先级 snapshot / _acquire_slot_possibly_preempting                      
  全部不激活；generation / env 拆开按信号量调度，但不会有任何 HP 抢 LP 的行为。                                                                  
                                                                                                                                                 
  场景 B：允许抢占                                                                                                                               
                  
  公共 wiring + 再加：                                                                                                                           
  - enable_priority_based_scheduling: true
  - priority_scheduling_policy: "fcfs" (目前只支持这一种，不填也默认 fcfs)                                                                       
                                                                          
  此时 AgenticExecutor.run() 会 build SchedState、向 data_coordinator 拉 initial snapshot + 注册 priority broadcast，HP 样本 env 回来时调用      
  _acquire_slot_possibly_preempting 去 abort 最新的 LP。                                                                                         
                                                                                                                                                 
  依赖前提: enable_priority_based_scheduling=true 要求 enable_action_level_scheduling=true (spec 里这样写的；单独开后者不会报错但也没意义)
```
