

# Requirements

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
```
source ./.venv/bin/activate
```

Install requirements with
```
pip install -r requirements.txt
```