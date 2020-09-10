# Federated Named Entity Recognition

## Setup 

Create a python virtualenv and activate it

```
python3 -m venv venv
source venv/bin/active
```

Install dependencies

```
pip install -r requirements.txt
```

:warning: PySyft 0.2.8 has a bug in the `model_centric_fl_client.py` file that breaks the communication with the Grid. Be sure to update the file in your environment site packages to the [current master version of the file](https://github.com/OpenMined/PySyft/blob/master/syft/grid/clients/model_centric_fl_client.py).

## Hosting the training plan

Launch PyGrid via docker-compose

```
docker-compose up -d
```

Create the training plan and host it on PyGrid

```
python host_plan.py
```

Test if plan is hosted properly

```
python test_plan.py
```

## Training

Use the a worker to execute the plan and run a training procedure.


