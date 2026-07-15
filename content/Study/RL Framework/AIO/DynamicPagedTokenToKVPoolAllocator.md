```py title="DynamicPagedTokenToKVPoolAllocator"
  ---                                                                                                                                            
  完整逻辑图                                                                                                                                     
                                                                                                                                                 
  ╔══════════════════════════════════════════════════════════════════════╗                                                                       
  ║           DynamicPagedTokenToKVPoolAllocator 完整架构               ║                                                                        
  ╚══════════════════════════════════════════════════════════════════════╝                                                                       
                                                                                                                                                 
  ┌─ 初始状态 ──────────────────────────────────────────────────────────┐                                                                        
  │  high_current_slot_width = layer_num  (全量层, 零额外开销)           │                                                                       
  │  low_current_slot_width  = layer_num  (全量层, 零额外开销)           │                                                                       
  │  所有优先级共享 free_pages, 无静态上限                               │                                                                       
  └─────────────────────────────────────────────────────────────────────┘                                                                        
                                                                                                                                                 
  ═══════════════════════ 分配入口 ═══════════════════════════════════════                                                                       
                                                                                                                                                 
  alloc_extend(prefix_lens, seq_lens, ..., is_high_priority: Tensor[bs])                                                                         
    │                                                       
    ▼                                                                                                                                            
  _resolve_priority(is_high_priority, bs) → per-req bool Tensor                                                                                  
    │                                                                                                                                            
    ├─ 全部 high ──────────────────────────── 单次路径 (priority_code=1)                                                                         
    ├─ 全部 low  ──────────────────────────── 单次路径 (priority_code=2)                                                                         
    └─ 混合 ───┐                                                                                                                                 
               ▼                                                                                                                                 
          拆分为 high_group / low_group                                                                                                          
          分别计算 high_pages_needed / low_pages_needed                                                                                          
               │                                                                                                                                 
               ▼                                                                                                                                 
                                                                                                                                                 
  ═══════════════════ 背压检查 (每组各一次) ════════════════════════════                                                                         
                                                                                                                                                 
  ensure_capacity(pages_needed, priority_code)                                                                                                   
    │                                                       
    ▼                                                                                                                                            
    while free_pages 不够:                                                                                                                       
    ┌──────────────────────────────────────────────────────────────────┐                                                                         
    │                     _try_backpressure()                         │                                                                          
    │                                                                 │                                                                          
    │  ① low_used > low_max_pages  且  low_width > min?              │                                                                           
    │     → _halve_and_repack(2, low_max_pages)                      │                                                                           
    │       全量迁移: 所有超出 low_max_pages 的页都跨页合并释放         │                                                                        
    │                                                                 │                                                                          
    │  ② high_used > high_reserved  且  high_width > min?            │                                                                           
    │     → _halve_and_repack(1, high_reserved_pages)                │                                                                           
    │       部分迁移: 只迁移超出 high_reserved_pages 的页              │                                                                         
    │       reserved 以内的页仅释放 partial slots                      │                                                                         
    │                                                                 │                                                                          
    │  ③ low_width > min? → 强制 _halve_and_repack(2, low_max_pages) │                                                                           
    │  ④ high_width > min? → 强制 _halve_and_repack(1, high_reserved)│                                                                           
    │  ⑤ return False (真的没有可以做的了)                              │                                                                        
    └──────────────────────────────────────────────────────────────────┘                                                                         
                                                                                                                                                 
  ═══════════════════ _halve_and_repack 详细 ═══════════════════════════                                                                         
                                                                                                                                                 
  _halve_and_repack(priority_code, max_keep_pages)                                                                                               
    │                                                                                                                                          
    ├─ Phase A: 页内 halving (_halve_slot_width)                                                                                                 
    │   │                                                                                                                                        
    │   │  每个页:                                                                                                                               
    │   │  slot_width: W → W/2                                                                                                                   
    │   │  保留偶数层 (0,2,4,...) 压缩到连续位置 (0,1,2,...)                                                                                     
    │   │  释放出 second sub-slot → partial_pool                                                                                                 
    │   │                                                                                                                                        
    │   │  例 (layer_num=8, W=8→4):                                                                                                              
    │   │  [L0 L1 L2 L3 L4 L5 L6 L7] → [L0 L2→1 L4→2 L6→3 | free free free free]                                                                 
    │   │                                 ═══ occupied ═══     ═══ partial ═══                                                                   
    │   ▼                                                                                                                                        
    │                                                                                                                                            
    ├─ Phase B: 跨页合并 (仅对超出 max_keep_pages 的部分)                                                                                        
    │   │                                                                                                                                        
    │   │  excess = current_used - max_keep_pages                                                                                                
    │   │  配对 excess/2 对 (donor, receiver):                                                                                                   
    │   │                                                                                                                                        
    │   │  Before:                                                                                                                               
    │   │  Receiver [occupied@ss=1 | free@ss=5]                                                                                                  
    │   │  Donor    [occupied@ss=1 | free@ss=5]                                                                                                  
    │   │                                                                                                                                        
    │   │  KV迁移: donor.slot1 的数据 → receiver.slot2 的位置                                                                                    
    │   │  for k in 0..new_width-1:                                                                                                              
    │   │    k_buf[r_ss-1+k][receiver_locs] = k_buf[d_ss-1+k][donor_locs]                                                                        
    │   │                                                                                                                                        
    │   │  After:                                                                                                                                
    │   │  Receiver [occupied@ss=1 | occupied@ss=5] ← 满了                                                                                       
    │   │  Donor    [empty] [empty]                  ← 释放回 free_pages                                                                         
    │   ▼                                                                                                                                        
    │                                                                                                                                            
    └─ Phase C: 更新 req_to_token_pool                                                                                                           
        │                                                                                                                                        
        │  构建 lookup table:                                                                                                                    
        │  remap[donor_encoded_locs] = receiver_encoded_locs                                                                                     
        │  req_to_token_pool[:] = remap[req_to_token_pool[:]]                                                                                    
        │                                                                                                                                        
        │  清理: donor 页 → priority=0, slot_width=0, 回到 free_pages                                                                            
        ▼                                                                                                                                        
                                                                                                                                                 
  ═══════════════════ 实际分配 ═══════════════════════════════════════                                                                           
                                                                                                                                                 
  _allocate_pages(num_pages, priority_code)                                                                                                      
    │                                                       
    ├─ 先消费 partial_pool 中的空闲 slots                                                                                                        
    ├─ 不够则从 free_pages 取新页 (无上限)                                                                                                       
    └─ 返回 (page_ids, slot_starts)                                                                                                              
                                                                                                                                                 
    ↓ (混合优先级时)                                                                                                                             
                                                                                                                                                 
  high_combined = high_pages + (high_ss - 1) * (num_pages + 1)                                                                                   
  low_combined  = low_pages  + (low_ss  - 1) * (num_pages + 1)
    │                                                                                                                                            
    ├─ reorder: [high_reqs..., low_reqs...]                                                                                                      
    ├─ combined: [high_combined, low_combined]                                                                                                   
    ├─ Triton kernel (alloc_extend_kernel / alloc_decode_kernel)                                                                                 
    └─ un-reorder: 结果恢复为原始 req 顺序                                                                                                       
    │                                                                                                                                            
    ▼                                                                                                                                            
  return (out_indices, dynamic_indices)                                                                                                          
                                                                                                                                                 
                                                                                                                                                 
  ═══════════════════ 高/低优先级对比 ═════════════════════════════════                                                                          
                                                                                                                                                 
                │    低优先级 (code=2)    │    高优先级 (code=1)     │                                                                           
  ─────────────┼────────────────────────┼─────────────────────────┤
   初始slot_width│    layer_num           │    layer_num             │                                                                           
   阈值          │    low_max_pages       │    high_reserved_pages   │                                                                           
   超预算时      │    halve + 全量repack  │    halve + 部分repack    │                                                                           
                │    (所有超出的页都合并   │    (只合并超出reserved    │                                                                         
                │     释放回free_pages)   │     的页, reserved以内    │                                                                          
                │                        │     只释放partial slots)  │                                                                           
   原因          │    数据量小, 全搬合算   │    数据量大, 只搬超出部分 │                                                                         
   最小slot_width│    2                   │    2                     │ 
```
