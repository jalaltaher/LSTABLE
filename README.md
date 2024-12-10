# LSTABLE library

LSTABLE is a Python library designed to sample trajectories and calculate theoretical features (density,characteristic function...) and display
- Stable distributions.
- $-\alpha$stable Lévy process.
- Classical Tempered Stable (CTS) Lévy processes.
---

## Installation

```bash
pip install LSTABLE

```


## Code Environment

To use the Levy library effectively, ensure you have the following environment setup:

- **Python Version**: Python 3.8 or newer
- **Dependencies**:
  - `numpy` (Numerical computations)
  - `matplotlib` (Visualizations)
  - `scipy` (Scientific computations)

Install the required dependencies using the following command:

```bash
pip install numpy matplotlib scipy
```

## Examples

### Stable Parameters 

The following examples demonstrate how to check the validity of the stable parameters and how to convert from the $1-$parameterization to the Lévy parameters 

### Sampling stable distribution

The following example demonstrates how to sample an $S_\alpha(\sigma,\beta,\mu)$

```python

# Parameters
alpha= 1.5 #stability index
sigma= 2.0 #
beta= 0.5 #
mu= 0.0 # 
n_sample=10000 # length of the sample

sample = stable_distribution_generator(
	alpha, sigma, beta, mu, n_sample
	)


# Compute the density
grid= np.linspace(-7,7,1000)
density = stable_density(
	grid ,alpha , sigma, beta, mu
	)
```

### Output Figure

- **Sample Stable Distributions**:  
  ![alpha=0.5](./figures/stable_hist_density_alpha05.png)
  ![alpha=1.5](./figures/stable_hist_density_alpha15.png)
  
  *This figure visualizes a histogram of 10000 values of $S_\alpha(\sigma,\beta,\mu)$ with alpha=0.5,1.5 and the corresponding density function computed using a Fourier Inverse formula.*



