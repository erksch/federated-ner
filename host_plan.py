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

from model import Net
from training_plan import build_training_plan
from averaging_plan import build_averaging_plan

model = Net()
model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter

training_plan = build_training_plan(model_params)
avg_plan = build_averaging_plan(model_params)

# PyGrid Node address
gridAddress = "127.0.0.1:5000"
grid = ModelCentricFLClient(id="test", address=gridAddress, secure=False)
print("Connecting to grid...")
grid.connect()# These name/version you use in worker
print("Success!")
name = "mnist"
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
        client_plans={'training_plan': training_plan},
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config
    )
    print("Host response:", response)
except GridError as e:
    print("Hosting failed: ", e)
