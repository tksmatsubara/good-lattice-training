# Number Theoretic Accelerated Learning of Physics-Informed Neural Networks (AAAI 2025)

Physics-informed neural networks solve partial differential equations by training neural networks. Since this method approximates infinite-dimensional PDE solutions with finite collocation points, minimizing discretization errors by selecting suitable points is essential for accelerating the learning process. Inspired by number theoretic methods for numerical analysis, we introduce good lattice training and periodization tricks, which ensure the conditions required by the theory. Our experiments demonstrate that GLT requires 2-7 times fewer collocation points, resulting in lower computational cost, while achieving competitive performance compared to typical sampling methods.

## PINNs

### Dependencies

- python v3.7.16
- numpy v1.24.3
- scipy v1.10.1
- matplotlib v3.7.1
- tensorflow v1.15.5
- protobuf v3.20.3
- pyDOE v0.3.8

Please download the datasets from the [official repository](https://github.com/maziarraissi/PINNs) and place them in `main/Data` directory.

All codes in `main` directory were made by modifying the code in the above repository.

### Command

Go `main/PINNsGLT` and run the command:

```sh
python PDE1D_GLT.py --dataset ${DATASET} --lattice ${LATTICE} --strategy ${STRATEGY} --sample ${SAMPLE} --nf ${NF}
python Poission_GLT.py -dataset ${DATASET} --s ${S} --lattice ${LATTICE} --strategy ${STRATEGY} --sample ${SAMPLE} --nf ${NF}
python PDE1D_GLT.py --identification --n_obs ${N_OBS} --dataset ${DATASET} --lattice ${LATTICE} --strategy ${STRATEGY} --sample ${SAMPLE} --nf ${NF}
```

Options:

- DATASET: `Poission_GLT.py` for Poission dataset and `PDE1D_GLT.py` for others
  - NLS
  - KdV
  - AC
  - Poisson
- LATTICE: method to determine collocation points
  - uni: Uniformly random sampling
  - sq: Uniformly spaced sampling
  - lhs: Latin Hypercube sampling
  - sob: Sobol Sequence
  - glt: Good lattice training (proposed)
- STRATEGY: optimizer to train PINNs
  - default: the L-BFGS-B method preceded by the Adam optimizer
  - cosine: Adam optimizer with cosine decay of a single cycle to zero
  - adam: Adam optimizer with no decay
- SAMPLE: lattice
  - fixed: use the same collocation points
  - random: get a different set of collocation points at every iteration
- NF: the number of collocation points
- S: the number of dimensions. Available only for Poission_GLT.py
- identification: perform system identifications. Available only for PDE1D_GLT.py
- N_OBS: the number of observations for system identification

See the PINNs' [official repository](https://github.com/maziarraissi/PINNs) for other options.

Examples:

```sh
python PDE1d_GLT.py --dataset NLS --lattice glt --strategy cosine --sample random --nf 610
python Poisson_GLT.py --s 2 --lattice glt --strategy default --sample fixed --nf 1597
```

Results:

Results are stored in `main/PINNsGLT/results`.

## CPINNs

### Dependencies

- python v3.9.16
- numpy v1.24.3
- scipy v1.10.1
- matplotlib v3.7.1
- pandas v1.5.3
- pytorch v1.13.1
- torchvision v0.14.1
- torchaudio v0.13.1
- pytorch-cuda v11.7
- pyDOE v0.3.8
- CGDs v0.4.5

Please download the code from the official [openreview.net page](https://openreview.net/forum?id=z9SIj-IM7tn) and copy the `Schrodinger` and `Burger` directories to it.

All codes in `Schrodinger` and `Burger` directories were made by modifying the code in the above openreview.net page.

### Command

At the root directory, run the command:

```sh
python Schrodinger/SchrodingerTrainGLT.py -lrmin 0.001 -lrmax 0.001 -pinn 4 100 -dis 4 200 --lattice ${LATTICE} --nf ${NF}
python Burger/BurgerTrainGLT.py -lrmin 0.001 -lrmax 0.001 -disLayer 8 -disNeu 60 -pinnLayer 8 -pinnNeu 60 --lattice ${LATTICE} --nf ${NF}
```

Options (same as above if available):

- LATTICE: method to determine collocation points
- NF: the number of collocation points

See the CPINNs' [openreview.net page](https://openreview.net/forum?id=z9SIj-IM7tn) for other options.

Examples:

```sh
python Schrodinger/SchrodingerTrainGLT.py -lrmin 0.001 -lrmax 0.001 -pinn 4 100 -dis 4 200 --lattice glt --nf 2584
python Burger/BurgerTrainGLT.py -lrmin 0.001 -lrmax 0.001 -disLayer 8 -disNeu 60 -pinnLayer 8 -pinnNeu 60 --lattice glt --nf 2584
```

Results:

Results are stored in `Schrodinger/output` or `Burger/output`.

## Reference

```bibtex
@inproceedings{Matsubara2025,
  title = {Number Theoretic Accelerated Learning of Physics-Informed Neural Networks},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  author    = {Matsubara, Takashi and Yaguchi, Takaharu},
  year = {2025},
}
```
