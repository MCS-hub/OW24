# Non-geodesically-convex optimization in the Wasserstein space

We leverage the code from https://github.com/PetrMokrov/Large-Scale-Wasserstein-Gradient-Flows for the JKO computation.

## Prerequisites
We recommend using Python version 3.8.0.

The list of required Python libraries is in `./requirements.txt`.

```bash
pip install -r requirements.txt
```

## Experiments

The main Python file is `script_complete.py`, experiment configs are in `./configs/` directory.
The directory `./archived/` contains experiment results used in the manuscript.

To run the Gaussian mixture experiment:

```bash
python script_complete.py conv_comp_dim_2 --discretization [discretization] --device [device]
```
where discretization is either fb or semi_fb. 

To run the von Mises-Fisher experiment:
```bash
python script_complete.py relaxed_vmF --discretization [discretization] --device [device]
```
where discretization is either fb or semi_fb. 

**Example**
```bash
python script_complete.py relaxed_vmF --discretization semi_fb --device cpu
```
