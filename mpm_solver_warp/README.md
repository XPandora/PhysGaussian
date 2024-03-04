# WARP MPM
<p align="center">
  <img src="assets/sand.gif">
</p>

This MPM solver is implemented using Nvidia's WARP: https://nvidia.github.io/warp/

For details about MPM, please refer to the course on the Material Point Method: https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf


## Prerequisites

This codebase is tested using the environment with the following key packages:

- Ubuntu 20.04
- CUDA 11.6
- Python 3.9.13
- Warp 0.10.1

## Installation
```
pip install -r requirements.txt
```

## Examples
Sand: column collapse 
```
python run_sand.py
```

More coming.