# 概述
本文档详解Sglang中为Deepseek V4架构提出的 Unified RadixCache 的tree cache结构。
从整体架构上来讲，UnifiedRadixCache的提出主要是为了统一不同的tree cache，似的不需要额外提供不同的tree cache的mixin，而是统一为full attention, sliding window attention, Mamba attention作为**pluggable components**存在于UnfiedRadixCache中。
```py title="UnifiedRadixCache"
┌───────────────────────────────────────────────┐
│              UnifiedRadixCache                │
│            (unified_radix_cache.py)           │
│                                               │
│  root_node ──► UnifiedTreeNode (radix tree)   │
│  components ► {ComponentType → TreeComponent} │
│  lru_lists ─► {ComponentType → UnifiedLRUList}│
└──────────┬───────────┬───────────┬────────────┘
           │           │           │
           ▼           ▼           ▼
   ┌────────────┐ ┌──────────┐ ┌─────────────┐
   │    Full    │ │   SWA    │ │    Mamba    │
   │ Component  │ │Component │ │  Component  │
   └─────┬──────┘ └────┬─────┘ └──────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
               ┌──────────────┐
               │TreeComponent │
               │    (ABC)     │
               └──────────────┘
```
# Code walkthrough
