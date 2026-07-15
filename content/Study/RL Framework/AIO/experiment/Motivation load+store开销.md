部署qwen3-30-A3B model，forward仅放一层，输入以10k为例，90%命中率

# bsz=64
1) 重算，既无命中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     64        
Benchmark duration (s):                  6.72      
Total input tokens:                      689811    
Total input text tokens:                 689811    
Total input vision tokens:               0         
Total generated tokens:                  64        
Total generated tokens (retokenized):    64        
Request throughput (req/s):              9.53      
Input token throughput (tok/s):          102723.18 
Output token throughput (tok/s):         9.53      
Total token throughput (tok/s):          102732.71 
Concurrency:                             34.73     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   3643.95   
Median E2E Latency (ms):                 3431.72   
---------------Time to First Token----------------
Mean TTFT (ms):                          3643.94   
Median TTFT (ms):                        3431.71   
P99 TTFT (ms):                           6631.43   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
2) 从host_memory中load/store
load_span_ms         mean=34.446  p50=35.108  p99=35.868
store_span_ms        mean=34.375  p50=35.040  p99=35.784
e2e_ms               mean=35.254  p50=35.928  p99=36.720
load_active_sum_ms   mean=22.532  p50=22.656  p99=25.159
store_active_sum_ms  mean=21.729  p50=21.972  p99=24.302
load_bw_gib_s        mean=6.18  p50=6.01  p99=9.28
store_bw_gib_s       mean=6.19  p50=6.02  p99=9.38
duplex_bw_gib_s      mean=12.07  p50=11.74  p99=18.14
batch_makespan_ms    mean=233.095  p50=233.095  p99=233.095
batch_duplex_bw_gib_s mean=115.83  p50=115.83  p99=115.83
实际可考虑到
3) 全部驻留在HBM中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     64        
Benchmark duration (s):                  1.71      
Total input tokens:                      685439    
Total input text tokens:                 685439    
Total input vision tokens:               0         
Total generated tokens:                  64        
Total generated tokens (retokenized):    64        
Request throughput (req/s):              37.42     
Input token throughput (tok/s):          400765.71 
Output token throughput (tok/s):         37.42     
Total token throughput (tok/s):          400803.13 
Concurrency:                             60.40     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1614.17   
Median E2E Latency (ms):                 1610.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          1614.17   
Median TTFT (ms):                        1610.98   
P99 TTFT (ms):                           1671.31   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     64        
Benchmark duration (s):                  3.90      
Total input tokens:                      688339    
Total input text tokens:                 688339    
Total input vision tokens:               0         
Total generated tokens:                  16384     
Total generated tokens (retokenized):    15430     
Request throughput (req/s):              16.42     
Input token throughput (tok/s):          176574.38 
Output token throughput (tok/s):         4202.86   
Total token throughput (tok/s):          180777.25 
Concurrency:                             63.59     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   3873.03   
Median E2E Latency (ms):                 3871.52   
---------------Time to First Token----------------
Mean TTFT (ms):                          1327.44   
Median TTFT (ms):                        1315.26   
P99 TTFT (ms):                           1385.15   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.98      
Median TPOT (ms):                        10.07     
P99 TPOT (ms):                           10.08     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.98      
Median ITL (ms):                         9.56      
P95 ITL (ms):                            9.84      
P99 ITL (ms):                            10.05     
Max ITL (ms):                            102.86    
==================================================
```
![[img_v3_0210v_ac290f5a-4acb-431a-ad63-42f7c076ce3g.jpg]]
# bsz=32
1) 重算，既无命中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     32        
Benchmark duration (s):                  3.37      
Total input tokens:                      344881    
Total input text tokens:                 344881    
Total input vision tokens:               0         
Total generated tokens:                  32        
Total generated tokens (retokenized):    32        
Request throughput (req/s):              9.49      
Input token throughput (tok/s):          102300.13 
Output token throughput (tok/s):         9.49      
Total token throughput (tok/s):          102309.62 
Concurrency:                             17.45     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1838.72   
Median E2E Latency (ms):                 1749.24   
---------------Time to First Token----------------
Mean TTFT (ms):                          1838.71   
Median TTFT (ms):                        1749.23   
P99 TTFT (ms):                           3312.94   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
2) 从host_memory中load/store
```py title=""
batch_makespan_ms     mean=215.377  p50=215.377  p99=215.377
batch_duplex_bw_gib_s mean=250.72  p50=250.72  p99=250.72
```
2) 全部驻留在HBM中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     32        
Benchmark duration (s):                  0.91      
Total input tokens:                      346632    
Total input text tokens:                 346632    
Total input vision tokens:               0         
Total generated tokens:                  32        
Total generated tokens (retokenized):    32        
Request throughput (req/s):              35.20     
Input token throughput (tok/s):          381280.93 
Output token throughput (tok/s):         35.20     
Total token throughput (tok/s):          381316.12 
Concurrency:                             29.29     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   832.17    
Median E2E Latency (ms):                 824.90    
---------------Time to First Token----------------
Mean TTFT (ms):                          832.16    
Median TTFT (ms):                        824.89    
P99 TTFT (ms):                           885.53    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
![[Pasted image 20260421095313.png]]
# bsz=16
1) 重算，既无命中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     16        
Benchmark duration (s):                  1.67      
Total input tokens:                      172338    
Total input text tokens:                 172338    
Total input vision tokens:               0         
Total generated tokens:                  16        
Total generated tokens (retokenized):    16        
Request throughput (req/s):              9.59      
Input token throughput (tok/s):          103294.50 
Output token throughput (tok/s):         9.59      
Total token throughput (tok/s):          103304.09 
Concurrency:                             9.01      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   939.31    
Median E2E Latency (ms):                 910.83    
---------------Time to First Token----------------
Mean TTFT (ms):                          939.30    
Median TTFT (ms):                        910.82    
P99 TTFT (ms):                           1639.29   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
2) 从host_memory中load/store
```py title=""
batch_makespan_ms     mean=109.506  p50=109.506  p99=109.506
batch_duplex_bw_gib_s mean=246.56  p50=246.56  p99=246.56
```
2) 全部驻留HBM
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     16        
Benchmark duration (s):                  0.55      
Total input tokens:                      172293    
Total input text tokens:                 172293    
Total input vision tokens:               0         
Total generated tokens:                  16        
Total generated tokens (retokenized):    16        
Request throughput (req/s):              29.31     
Input token throughput (tok/s):          315664.04 
Output token throughput (tok/s):         29.31     
Total token throughput (tok/s):          315693.35 
Concurrency:                             12.75     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   435.05    
Median E2E Latency (ms):                 415.10    
---------------Time to First Token----------------
Mean TTFT (ms):                          435.05    
Median TTFT (ms):                        415.09    
P99 TTFT (ms):                           527.57    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 16        
Successful requests:                     256       
Benchmark duration (s):                  34.08     
Total input tokens:                      2770121   
Total input text tokens:                 2770121   
Total input vision tokens:               0         
Total generated tokens:                  65536     
Total generated tokens (retokenized):    61134     
Request throughput (req/s):              7.51      
Input token throughput (tok/s):          81278.96  
Output token throughput (tok/s):         1922.91   
Total token throughput (tok/s):          83201.87  
Concurrency:                             15.95     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2123.21   
Median E2E Latency (ms):                 2046.53   
---------------Time to First Token----------------
Mean TTFT (ms):                          534.14    
Median TTFT (ms):                        443.33    
P99 TTFT (ms):                           1149.65   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.23      
Median TPOT (ms):                        6.22      
P99 TPOT (ms):                           7.15      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           6.23      
Median ITL (ms):                         5.93      
P95 ITL (ms):                            6.35      
P99 ITL (ms):                            6.81      
Max ITL (ms):                            471.07    
==================================================
```
![[img_v3_0210v_6b706dec-6fba-4e61-a1c9-7c0fd77fbe9g.jpg]]
# bsz=8
1) 重算，既无命中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     8         
Benchmark duration (s):                  0.84      
Total input tokens:                      86190     
Total input text tokens:                 86190     
Total input vision tokens:               0         
Total generated tokens:                  8         
Total generated tokens (retokenized):    8         
Request throughput (req/s):              9.48      
Input token throughput (tok/s):          102167.27 
Output token throughput (tok/s):         9.48      
Total token throughput (tok/s):          102176.75 
Concurrency:                             4.70      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   495.14    
Median E2E Latency (ms):                 484.94    
---------------Time to First Token----------------
Mean TTFT (ms):                          495.13    
Median TTFT (ms):                        484.93    
P99 TTFT (ms):                           823.00    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```
2) 从host_memory中load/store
```py title=""
batch_makespan_ms    mean=81.669  p50=81.669  p99=81.669
batch_duplex_bw_gib_s mean=82.65  p50=82.65  p99=82.65
```
2) 全部驻留在HBM中
```py title=""
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     8         
Benchmark duration (s):                  0.29      
Total input tokens:                      85752     
Total input text tokens:                 85752     
Total input vision tokens:               0         
Total generated tokens:                  8         
Total generated tokens (retokenized):    8         
Request throughput (req/s):              27.94     
Input token throughput (tok/s):          299517.17 
Output token throughput (tok/s):         27.94     
Total token throughput (tok/s):          299545.11 
Concurrency:                             6.88      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   246.19    
Median E2E Latency (ms):                 246.09    
---------------Time to First Token----------------
Mean TTFT (ms):                          246.18    
Median TTFT (ms):                        246.08    
P99 TTFT (ms):                           273.03    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P95 ITL (ms):                            0.00      
P99 ITL (ms):                            0.00      
Max ITL (ms):                            0.00      
==================================================
```