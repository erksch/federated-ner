import syft as sy
import torch as th
import torch.nn as nn
from model import Net

@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

def build_averaging_plan(model_params):
    # Build the Plan
    print("Building averaging plan...")
    _ = avg_plan.build(model_params, model_params, th.tensor([1.0]))
    print("Done.")

    # Let's check Plan contents
    #print(avg_plan.code)

    return avg_plan
