import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.grid.exceptions import GridError
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

import os
import websockets
import json
import requests
import torch as th
import torch.nn as nn

from model import Net

sy.make_hook(globals())
# force protobuf serialization for tensors
hook.local_worker.framework = None
th.random.manual_seed(1)

EMBEDDING_DIM = 100

model = Net(EMBEDDING_DIM)
model_params = [param.data for param in model.parameters()]

def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

def softmax_cross_entropy_with_logits(logits, targets, batch_size, loss_weights):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs * loss_weights).sum() / batch_size

def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad

@sy.func2plan()
def training_plan(X, y, batch_size, lr, loss_weights, model_params):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size, loss_weights)

    # backprop
    loss.backward()

    # step
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model_params
    ]
    
    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        *updated_params
    )

# Dummy input parameters to make the trace
B = 10
X = th.randn(B, EMBEDDING_DIM)
y = nn.functional.one_hot(th.Tensor(B).random_(0, 4).long(), 5)
loss_weights = th.randn(5)
lr = th.tensor([0.01])
batch_size = th.tensor([B])
    
print("Building training plan...")
_ = training_plan.build(X, y, batch_size, lr, loss_weights, model_params, trace_autograd=True)
print("Done.")

@sy.func2plan()
def eval_plan(X, y, model_params):
    with th.no_grad():
        model.eval()

        # inject params into model
        set_model_params(model, model_params)

        # forward pass
        logits = model.forward(X)
        
        pred = th.argmax(logits, dim=1)
        target = th.argmax(y, dim=1)

        return (pred, target)

X = th.randn(B, EMBEDDING_DIM)
y = nn.functional.one_hot(th.Tensor(B).random_(0, 4).long(), 5)

print("Building evaluation plan...")
_ = eval_plan.build(X, y, model_params, trace_autograd=True)
print("Done.")

@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

print("Building averaging plan...")
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))
print("Done.")

# PyGrid Node address
gridAddress = "127.0.0.1:5000"
grid = ModelCentricFLClient(id="test", address=gridAddress, secure=False)
print("Connecting to grid...")
grid.connect()# These name/version you use in worker
print("Success!")
name = "conll-100d"
version = "1.0.0"

client_config = {
    "name": name,
    "version": version,
    "batch_size": 64,
    "lr": 0.005,
    "max_updates": 100  # custom syft.js option that limits number of training loops per worker
}

server_config = {
    "min_workers": 5,
    "max_workers": 5,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 5,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True,  # tells PyGrid that avg plan is executed per diff
    #"authentication": {
    #    "type": "jwt",
    #    "pub_key": public_key,
    #}
}

model_params_state = State(
    state_placeholders=[
        PlaceHolder().instantiate(param)
        for param in model_params
    ]
)

print("Hosting plan...")

try:
    response = grid.host_federated_training(
        model=model_params_state,
        client_plans={
            'training_plan': training_plan,
            'eval_plan': eval_plan
        },
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config
    )
    print("Host response:", response)
except GridError as e:
    print("Hosting failed: ", e)
