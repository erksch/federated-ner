import syft as sy
import torch as th
import torch.nn as nn
from model import Net

sy.make_hook(globals())
# force protobuf serialization for tensors
hook.local_worker.framework = None
th.random.manual_seed(1)

model = Net()

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

def softmax_cross_entropy_with_logits(logits, targets, batch_size):
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
    return -(targets * log_probs).sum() / batch_size

def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad

@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)

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

def build_training_plan(model_params):
    # Dummy input parameters to make the trace
    X = th.randn(3, 28 * 28)
    y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
    lr = th.tensor([0.01])
    batch_size = th.tensor([3.0])
      
    print("Building training plan...")
    _ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)
    print("Done.")

    #print(training_plan.code)
    #print(training_plan.torchscript.code)
