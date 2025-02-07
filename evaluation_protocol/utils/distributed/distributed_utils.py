import torch
import torch.distributed as dist
from datetime import timedelta
import os

def initialize_distributed():
    """
    Initializes the distributed environment
    """
    backend = 'nccl'
    torch.distributed.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=7200000),
    )
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)

def cleanup_distributed():
    """
    Cleans up the distributed environment
    """
    torch.distributed.destroy_process_group()

def print_r0(message: str, *args, **kwargs):
    """
    Prints a message in the rank 0 process
    :param message: The message to print
    """
    if dist.get_rank() == 0:
        print(message, *args, **kwargs)