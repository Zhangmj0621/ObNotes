megatron/training/arguments.py

```bash
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        args.rank = int(os.getenv('RANK', '0'))
        args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args
```

![[Pasted image 20260408145432.png]]