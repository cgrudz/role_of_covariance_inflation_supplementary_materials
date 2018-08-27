# role_of_covariance_inflation_supplementary_materials

This repository contains the essential scripts and supplementary materials for the manuscript 
[Chaotic dynamics and the role of covariance inflation for reduced rank Kalman filters with model error](https://www.nonlin-processes-geophys-discuss.net/npg-2018-4/)

The two nonlinear experiments are given in ekf_v_ause_v_aus.py and aus_inflation_test.py; the former compares the performance of EKF-AUS, EKF-AUSE and EKF, the later tests the performance of EKF-AUS
with multiplicative inflation.

The script ekf_ause_riccati.py contains the computation of the KF-AUSE Riccati equation which can be used indepentently for linear experiments not included in this repository.  For the linear experiments,
one needs only to compute the Riccati equation directly from a sequence of propagators defined by the tangent linear model, as defined in l96.py.  The eigenvalues and BLV projection coefficients can
be computed directly from this Riccati equation as it is exact in a linear model.

Licence
------------------------------------------------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./licence.txt)

