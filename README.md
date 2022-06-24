# Optimal Switching with Jumps
Code repository for the Optimal Switching with Jumps (OSJ) algorithm described in [A neural network approach to high-dimensional optimal switching problems with jumps in energy markets](no link). This code is written as part of a Ph.D. research project by April Nellis in conjunction with advisors Dr. Erhan Bayraktar and Dr. Asaf Cohen. We investigate applications of deep-learning algorithms to optimal switching problems.

## Code File List:
- **Aid_LearnJumpSwitchML.py**: Multi-power-plant example inspired by the optimal switching problem described in [Aid et al.](https://epubs.siam.org/doi/abs/10.1137/120897298)

- **CL_LearnJumpSwitchML.py**: Adaptation of of Pham's [RDBDP algorithm](https://www.researchgate.net/publication/337746171_Deep_backward_schemes_for_high-dimensional_nonlinear_PDEs) for a d-dimensional jump-diffusion process representing natural gas and electricity prices for a power plant which is trying to predict the optimal production schedule under stochastic price fluctuations. Power plant model is taken from Carmona and Ludkovski's 2008 paper [PRICING ASSET SCHEDULING FLEXIBILITY USING OPTIMAL SWITCHING](https://www.tandfonline.com/doi/full/10.1080/13504860802170507).

- **JumpML.py**: Adaptation of of Pham's [RDBDP algorithm](https://www.researchgate.net/publication/337746171_Deep_backward_schemes_for_high-dimensional_nonlinear_PDEs) for a stock with underlying jump-diffusion price process, intended to test the learning ability of the loss function adjusted to accommodate a jump process. Algorithm is applied to the following option-pricing examples:
1. [Deep Backward Schemes for High-dimensional Nonlinear PDEs](https://www.researchgate.net/publication/337746171_Deep_backward_schemes_for_high-dimensional_nonlinear_PDEs), which does not involve a jump in the price process
2. [A penalty method for American options with jump diffusion processes](https://link.springer.com/article/10.1007/s00211-003-0511-8) by D'Halluin, Forsyth, and Labahn
3. [A Jump-Diffusion Model for Option Pricing](http://www.columbia.edu/~sk75/MagSci02.pdf) by Kou, later expanded upon in [Option Pricing Under a Double Exponential Distribution](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.1030.0163) by Kou and Wang

- **EnergyCLOrig.py**: Implementation of Longstaff-Schwartz algorithm involving regression over Monte Carlo paths, described in [Carmona, Ludkovski (2008)](https://www.tandfonline.com/doi/full/10.1080/13504860802170507), for comparison purposes.

- **visualsML.py**: Holds functions for visualizing output of various algorithms (imported to all other files)

## Instructions on running code:
Simply type `python [filename]`
- Figures will be saved in a Figures/ folder
- Animations will be saved in an Animations/ folder
- Neural network weights will be saved in a Weights/ folder
Users may have to manually create empty folders with these names before executing the code.
