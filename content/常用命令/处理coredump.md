ulimit -c unlimited 
sysctl -w kernel.core_pattern=/path/to/dumpdir/core.%e.%p.%h.%t 
set max-value-size unlimited 
cat /proc/sys/kernel/core_pattern