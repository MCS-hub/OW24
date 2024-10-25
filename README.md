# Non-geodesically-convex optimization in the Wasserstein space
This repository contains the source code and the experiment results of the paper:

>Hoang Phuc Hau Luu, Hanlin Yu, Bernardo Williams, Petrus Mikkola, Marcelo Hartmann, Kai Puolamäki, and Arto Klami. Non-geodesically-convex optimization in the Wasserstein space. _Advances in
Neural Information Processing Systems_, 2024.


We leverage the code from https://github.com/PetrMokrov/Large-Scale-Wasserstein-Gradient-Flows 
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

**FB Euler or semi FB Euler**

```bash
python script_complete.py --experiment conv_comp_dim_2 --method ICNN_jko_dc --discretization [discretization] --device [device]
```
where discretization is either fb or semi_fb. 

**ULA**

```bash
python script_complete.py --experiment conv_comp_dim_2 --method EM_sim_10000 --device [device]
```

To run the von Mises-Fisher experiment:

**FB Euler or semi FB Euler**

```bash
python script_complete.py --experiment relaxed_vmF --discretization [discretization] --device [device]
```
where discretization is either fb or semi_fb. 


**Example**
```bash
python script_complete.py --experiment relaxed_vmF --discretization semi_fb --device cpu
```
