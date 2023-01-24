# Optimal Switching with Jumps
Code repository for the Optimal Switching with Jumps (OSJ) algorithm described in [A neural network approach to high-dimensional optimal switching problems with jumps in energy markets](https://arxiv.org/abs/2210.03045). This code is written as part of a Ph.D. research project by April Nellis in conjunction with advisors Dr. Erhan Bayraktar and Dr. Asaf Cohen. We investigate applications of deep-learning algorithms to optimal switching problems.

## Code File List:
- **Aid_LearnJumpSwitchML.py**: Multi-power-plant example inspired by the optimal switching problem described in [Aid et al.](https://epubs.siam.org/doi/abs/10.1137/120897298)

- **CL_LearnJumpSwitchML.py**: Adaptation of of Pham's [RDBDP algorithm](https://www.researchgate.net/publication/337746171_Deep_backward_schemes_for_high-dimensional_nonlinear_PDEs) for a d-dimensional jump-diffusion process representing natural gas and electricity prices for a power plant which is trying to predict the optimal production schedule under stochastic price fluctuations. Power plant model is taken from Carmona and Ludkovski's 2008 paper [PRICING ASSET SCHEDULING FLEXIBILITY USING OPTIMAL SWITCHING](https://www.tandfonline.com/doi/full/10.1080/13504860802170507).

- **EnergyCLOrig.py**: Implementation of Longstaff-Schwartz algorithm involving regression over Monte Carlo paths, described in [Carmona, Ludkovski (2008)](https://www.tandfonline.com/doi/full/10.1080/13504860802170507), for comparison purposes.

- **visualsML.py**: Contains functions for visualizing output of various algorithms (imported to all other files)

## Instructions on running code:
Simply type `python [filename]` in command line.
- Figures will be saved in a Figures/ folder
- Animations will be saved in an Animations/ folder
- Neural network weights will be saved in a Weights/ folder

NOTE: Users may have to manually create empty folders with these names before executing the code.
