# Implementation of Nested Sampling on Light House Problem
By Ammar Zahin, Ioannis Michaloliakos, Nimesh Pokhrel, and Sanjib Katuwal

## Light house Problem Statement
#### 2D Problem
A lighthouse is somewhere off a piece of straight coastline at a position α along the shore and a distance β out at sea. It emits a series of short highly collimated flashes at random intervals and hence at random azimuths. These pulses are intercepted on the coast by photo-detectors that record only the fact that a flash has occurred, but not the angle from which it came. N flashes have so far been recorded at positions {x<sub>k</sub>}. Where is the lighthouse?’
![2d-light-house](images/2d_light_house.jpg)

#### 3D Problem
An analogous problem in 3D so that the lighthouse is now at a position (α, β, γ) and the flashes are emitted at random azimuths and polar angles. The photo-detectors are now located at positions {x<sub>k</sub>, y<sub>k</sub>}. Where is the lighthouse?’

## Bayesian Inference
Bayesian inference derives the posterior porbablity from two antecendents: a prior probability and a likelihood function. The posterior probability is the probability of a hypothesis given the data. The prior probability is the probability of a hypothesis before the data is observed. The likelihood function is the probability of the data given the hypothesis. The posterior probability is calculated using Bayes’ theorem:
$$P(H|E)=\frac{P(E|H)\cdot P(H)}{P(E)}$$
where, the LHS is the posterior probability, on RHS $P(E|H)$ is the likelihood function, $P(H)$ is the prior probability, and $P(E)$ is the evidence.
### Likelihood Function
#### 2D Case:
The lighthouse emits light in random directions,
$$ L(\theta)d\theta = \frac{1}{\pi}d\theta$$
In cartesian coordinates, the likelihood function is,
$$ L(\textbf{r})dr=\frac{1}{\pi}\frac{z_{LH}}{(x-x_{LH}^2)^2+z_{LH}^2}$$
where, subscript LH denotes the Light House.
#### 3D Case:
The lighthouse emits light in random directions,
$$L(\theta,\phi)d\theta d\phi=\frac{1}{\pi^2}d\theta d\phi$$
In cartesian coordinates, the likelihood function is,
$$L(\textbf{r})dr=\frac{1}{\pi^2}\frac{z_{LH}}{(x-x_{LH}^2)^2+(y-y_{LH}^2)^2+z_{LH}^2}\frac{1}{\sqrt{(x-x_{LH}^2)^2+(y-y_{LH}^2)^2}}$$
where, subscript LH denotes the Light House.

### Multiple Light Houses
The likelihoods for multiple lighthouses add up with weighted ratios equal to their relative brightness.
$$L(\textbf{r}_1,\textbf{r}_2)=I_1L(\textbf{r}_1)+I_2L(\textbf{r}_2)$$
where, $I_1$ and $I_2$ are the relative brightness of the two lighthouses and $I_1+I_2=1$.


## Nested Sampling
Nested Sampling is a computational method for Bayesian inference. It is a Monte Carlo method developed by physicist John Skilling in 2004.

## Code Architecture
![code-architecture](images/architecture.png)
## Results
### 2D Problem
#### One Light House
![2d-1light-house-dist](images/2D1LHW.png)
![2d-1light-house-corner](images/2D1LHC.png)

#### Three Light House
![2d-3light-house-dist](images/2D3LHW.png)
![2d-3light-house-corner](images/2D3LHC.png)

### 3D Problem
#### One Light House
![3d-1light-house-dist](images/3D1LHD.png)
![3d-1light-house-dist](images/3D1LHW.png)
![3d-1light-house-corner](images/3D1LHC.png)
![3d-1light-house-corner](images/3D1LH3D.png)

#### One Light House
![3d-2light-house-dist](images/3D2LHD.png)
![3d-2light-house-dist](images/3D2LHW.png)
![3d-2light-house-corner](images/3D2LHC.png)
![3d-2light-house-corner](images/3D2LH3D.png)


## References
1. [Nested Sampling by John Skilling](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full)
2. [Light House Problem](http://www.di.fc.ul.pt/~jpn/r/bugs/lighthouse.html)
3. [Sivia Data Analysis with Skilling](https://www.amazon.com/dp/0198568320)