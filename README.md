# mc-XY-model
Monte-Carlo simulation of the 2D XY model. This document includes multiple MC algorithms. Both local Metropolis updates and Wolff cluster updates have been implemented. An single code acts as the main thermalization + measurement of thermodynamical quantities, which are stored in saved numpy arrays. Then, another code analyses the data and return autocorrelation functions and error bars using the Jankknife method.

This code was written for Python 3.8.10. Package dependencies are
- joblib 1.1.0  (for parallelization)
- numba 0.55.1  (for optimization and C pre-compiler)
- numpy 1.21.5

The code is set such that it can be run in parallel, dispatching jobs to different cores within the same node. A parallel tempering sub-routine can be activated, which improved thermalization compared to serial runs.

