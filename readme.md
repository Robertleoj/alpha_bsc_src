

# Requirements

## DB
Run `init_db.sh`

## C++
Create a folder in `cpp_src` called `.libs`. 

### Torch 
Download torch binaries from [here](https://pytorch.org/) (around 2 GB) and extract them into `.libs`.

### GSL
Install the *GNU Scientific Library* (GSL). The library must be put in the `.libs/gsl` directory. Follow instructions from [here](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/).

## Python
Create a virtual environment with

```bash
python3 -m venv .venv
```

Activate it with
```bash
source ./.venv/bin/activate
```

Install requirements with

```bash
pip install -r requirements.txt
```

Also, the connect4 solver needs to be installed. To do so, run

```bash
python3 setup.py install
```

inside the `py_src/conn4_alpha_beta` folder with the virtual environment activated.

## Neural Network
Run `init_conn4_net.py`

# To train
Run `local_train_cycle.py`