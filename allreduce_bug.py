import os
import shutil
import tempfile
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet import DistributedStrategy

LABEL = 10
HIDDEN = 512

class MLPLayer(paddle.nn.Layer):

    def __init__(self, hidden, label):
        super(MLPLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(hidden, hidden)
        self.linear2 = paddle.nn.Linear(hidden, hidden)
        self.linear3 = paddle.nn.Linear(hidden, label)

        self.relu1 = paddle.nn.ReLU()
        self.relu2 = paddle.nn.ReLU()

    def forward(self, input_):
        output = self.linear1(input_)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output = self.linear3(output)

        return output

def create_strategy():

    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10

    build_strategy.fuse_bn_act_ops = True
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_add_act_ops = True
    build_strategy.enable_addto = True

    return build_strategy, exec_strategy

def get_dist_optimizer(optimizer):
    build_strategy, exec_strategy = create_strategy()

    dist_strategy = DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.fuse_all_reduce_ops = True
    all_reduce_size = 0
    dist_strategy.fuse_grad_size_in_MB = all_reduce_size
    dist_strategy.nccl_comm_num = 1
    cudnn_deterministic = False
    dist_strategy.cudnn_exhaustive_search = not cudnn_deterministic
    dist_strategy.conv_workspace_size_limit = 4096  # MB
    dist_strategy.cudnn_batchnorm_spatial_persistent = True
    dist_strategy.sync_nccl_allreduce = True
    dist_strategy.gradient_scale_configs = {'scale_strategy': 'sum'}

    dist_strategy.amp = True
    dist_strategy.amp_configs = {
        "init_loss_scaling": 128.0,
        "use_dynamic_loss_scaling": True,
        "use_pure_fp16": False
    }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

    return optimizer

def compile(program, loss_name=None):
    build_strategy, exec_strategy = create_strategy()

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program


def build(main_prog, startup_prog, need_acc=False):
    with paddle.static.program_guard(main_prog, startup_prog):
        with paddle.utils.unique_name.guard():
            input_holder = paddle.static.data(
                name="input_", shape=[None, HIDDEN], dtype="float32")
            label_holder = paddle.static.data(
                name="label_", shape=[None, 1], dtype="int64")

            model = MLPLayer(HIDDEN, LABEL)
            out = model(input_holder)
            loss = F.cross_entropy(out, label_holder)

            if need_acc:
                acc_top1 = paddle.metric.accuracy(input=out, label=label_holder, k=1)
                acc_top1_not_redue = paddle.metric.accuracy(input=out, label=label_holder, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label_holder, k=5)
                acc_top5_not_redue = paddle.metric.accuracy(input=out, label=label_holder, k=5)

                paddle.distributed.all_reduce(acc_top1, op=paddle.distributed.ReduceOp.SUM)
                acc_top1 = acc_top1 / paddle.distributed.get_world_size()
                paddle.distributed.all_reduce(acc_top5, op=paddle.distributed.ReduceOp.SUM)
                acc_top5 = acc_top5 / paddle.distributed.get_world_size()

                return acc_top1, acc_top5, acc_top1_not_redue, acc_top5_not_redue
            else:
                optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
                optimizer = get_dist_optimizer(optimizer)

                optimizer.minimize(loss, startup_prog)
                return None, None, None, None

def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = paddle.static.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = paddle.static.load_program_state(path)
    return state


def load_params(prog, path, ignore_params=None):


    print("Loading parameters from {}...".format(path))

    ignore_set = set()
    state = _load_state(path)

    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])

    if ignore_params:
        all_var_names = [var.name for var in prog.list_vars()]
        ignore_list = filter(
            lambda var: any([re.match(name, var) for name in ignore_params]),
            all_var_names)
        ignore_set.update(list(ignore_list))

    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                logger.warning(
                    'variable {} is already excluded automatically'.format(k))
                del state[k]

    paddle.static.set_program_state(prog, state)


paddle.enable_static()

fleet.init(is_collective=True)

device = paddle.set_device('gpu')

main_prog = paddle.static.Program()
eval_prog = paddle.static.Program()
startup_prog = paddle.static.Program()

build(main_prog, startup_prog, need_acc=False)
acc_top1, acc_top5, acc_top1_not_redue, acc_top5_not_redue = \
    build(eval_prog, startup_prog, need_acc=True)
eval_prog = eval_prog.clone(for_test=True)

compiled_eval_prog = compile(eval_prog)

exe = paddle.static.Executor(device)
exe.run(startup_prog)

np.random.seed(paddle.distributed.get_rank())
input_data = np.random.random((64, HIDDEN)).astype("float32")
label_data = np.random.randint(0, LABEL-1, (64, 1)).astype("int64")


exe.run(program=main_prog,
        feed={"input_": input_data, "label_": label_data})
load_params(main_prog, "./test_allreduce_mode/paddle_example")

acc_top1_reduced, acc_top5_reduced, acc_top1_nonreduced, acc_top5_nonreduced = exe.run(program=compiled_eval_prog,
        feed={"input_": input_data, "label_": label_data},
        fetch_list=[acc_top1.name, acc_top5.name,
                    acc_top1_not_redue.name, acc_top5_not_redue.name])
print("Top_1:", acc_top1_reduced, "Top_5:", acc_top5_reduced)
print("Top_1 (Non-reduced):", acc_top1_nonreduced, "Top_5 (Non-reduced):", acc_top5_nonreduced)
