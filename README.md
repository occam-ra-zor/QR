# Quantile Regulariation : Towards Implicit Calibration of Regression Models


Data links


| Dataset | N | D | link  |
| --- | --- |--- | --- |
|  Airfoil | 1503 | 5 |https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise |
|  Boston  | 506 | 13 | https://github.com/selva86/datasets/blob/master/BostonHousing.csv|
|  Concrete | 1030  | 8 |http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength  |
|  Fish Toxicity    | 908 | 6 | https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity|
|  Kin8nm   | 8182 | 8 |  https://www.openml.org/d/189 |
|  Protein Structure | 45730 |9 | https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure |
|  Red Wine | 1599 | 11 |  https://archive.ics.uci.edu/ml/datasets/wine+quality|
|  White Wine | 4898 | 11 |   https://archive.ics.uci.edu/ml/datasets/wine+quality |
|  Yacht Hydrodynamics | 308 | 6 |   http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics |
|  Year  Prediction MSD | 515345 | 90 |  https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD |



Sample experiments are provided  for both dropout-VI and ensemble-VI models

```
Dropout-VI on Airfoil  without Quantile Regularization
--------------------------------
calib         : 14.01 -+ 1.67                     
iso_calib     : 19.37 -+ 3.42
rmse          : 3.63 -+ 0.10
iso_rmse      : 3.63 -+ 0.10
nll           : 2.70 -+ 0.02
iso_nll       : -1.14 -+ 0.57
time          : 1.36 -+ 0.06
iso_time      : 0.08 -+ 0.01
iso nll count : 18 , maximum likelihood :12022.05078125
--------------------------------
```

```
Dropout-VI on Airfoil with Quantile Regularization (lambda = 1)
----------------------------------
calib         : 9.48 -+ 1.78
iso_calib     : 12.73 -+ 1.46
rmse          : 3.91 -+ 0.11
iso_rmse      : 3.91 -+ 0.11
nll           : 2.78 -+ 0.03
iso_nll       : -0.76 -+ 0.32
time          : 1.91 -+ 0.06
iso_time      : 0.08 -+ 0.01
iso nll count : 14 , maximum likelihood :4775.97607421875

----------------------------------
