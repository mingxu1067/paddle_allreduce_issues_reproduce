# Memory Access Issues of Allreduce
## Description
There is a memory accessing issue of allreduce operation when calling allreduce twice to two paddle variable. For example, calling allreduce to top1 and top5 accuracy in evaluation stage with 8 devices. We found that some processes would switch values of top1 and top5, like `allreduce_top1 = ReduceOp([top1 of device 0-5 and top5 of device 6-7]` and `allreduce_top5 = ReduceOp([top5 of device 0-5 and top1 of device 6-7]`.

We launched testing script twice with same data input and model weights, also fixed random seeds. But we got different reduced top1 and top5. In `log/workerlog.7`, we found that top1 and top5 were switched after allreduce, as the message showed below. See `log` for details.
```
W0214 01:43:37.128763 752003 device_context.cc:447] Please NOTE: device: 7, GPU Compute Capability: 8.0, Driver API Version: 11.4, Runtime API Version: 11.2
W0214 01:43:37.131966 752003 device_context.cc:465] device: 7, cuDNN Version: 8.2.
I0214 01:43:40.344686 752003 gen_comm_id_helper.cc:190] Server listening on: 127.0.0.1:47997 successful.
W0214 01:43:49.677826 752003 build_strategy.cc:110] Currently, fuse_broadcast_ops only works under Reduce mode.
I0214 01:43:49.679417 752003 fuse_pass_base.cc:57] ---  detected 2 subgraphs
I0214 01:43:49.679685 752003 fuse_pass_base.cc:57] ---  detected 2 subgraphs
W0214 01:43:49.681643 752003 fuse_all_reduce_op_pass.cc:76] Find all_reduce operators: 6. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 1.
Loading parameters from ./test_allreduce_mode/paddle_example...
I0214 01:43:50.820441 752003 fuse_pass_base.cc:57] ---  detected 2 subgraphs
Top_1: [0.484375] Top_5: [0.1796875]
Top_1 (Non-reduced): [0.09375] Top_5 (Non-reduced): [0.53125]
```

## Reproduce Steps
#### Requirement
- PaddlePaddle v2.2
- CUDA 11.4
- A100 GPUs
- Python3.7

#### Command to run
```Bash
$ git clone https://github.com/mingxu1067/paddle_allreduce_issues_reproduce.git
$ cd paddle_allreduce_issues_reproduce
$ python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 allreduce_bug.py
# Call multiple times, the accuracy would vary.
```
