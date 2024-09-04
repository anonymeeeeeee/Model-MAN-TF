# Model-MAN-TF

# Geometry-Aware Deep Learning for 3D Skeleton-Based Motion Prediction

This is the pytorch version, python code of our paper Geometry-Aware Deep Learning for 3D Skeleton-Based Motion Prediction. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/8989d3a4-c95f-47d3-9010-2448314dc678" alt="First Image" width="200"/>
  <img src="https://github.com/user-attachments/assets/92904789-6b56-4130-9ad4-af7fc217d1cb" alt="Second Image" width="200"/>
  <img src="https://github.com/user-attachments/assets/636bfbc6-473c-4748-861c-a2b5b34ed23b" alt="third Image" width="200"/>
</p>

## Overview

In this paper, we propose a novel approach, MAN-TF, which combines Kendall's shape space with Lie group representations and transformers for accurate body movement characterization, with a focus on improving long-term prediction accuracy. 
In doing so, we model both rigid and non rigid transformations of the skeletons involved in human motion for smooth and plausible predictions in long-term horizons.

![man_tf_overview](https://github.com/anonymeeeeeee/Model-MAN-TF/assets/161598974/fceb551b-24ac-49c8-a52f-271646eeaf9b)

## Datasets

* H3.6m dataset is a 3D human pose dataset containing 3.6 million human poses.
Please visit http://vision.imar.ro/human3.6m/ in order to request access and contact the maintainers of the dataset.

* To obtain the GTA-IM dataset please check: https://github.com/ZheC/GTA-IM-Dataset#requesting-dataset 

* To obtain the PROX dataset please check: https://prox.is.tue.mpg.de/


## Train and Test

* Run the following jupyter "Train&Test_Transformer_Kendal_Lie_Human" to train and test the model in H3.6m

* Run the following jupyter "Train&Test_Transformer_Kendall_Lie_Gta" to train and test the model in GTA-IM

* Run the following jupyter "Train&Test_Transformer_Kendall_Lie_Prox" to train and test the model in PROX

## MAE results for all actions 

H3.6M  | 80ms | 160ms | 320ms | 400ms | 560ms | 640ms | 720ms | 1000ms | 
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:  | :----: 
walking |  0.252 | 0.257 | 0.284  | 0.290 | 0.299 | 0.306 |  0.313|  0.329  |
eating | 0.203 | 0.212  |  0.218 | 0.219 | 0.224  | 0.225 |0.227| 0.233 |
smoking | 0.327 | 0.358 | 0.395 | 0.411 | 0.422 | 0.426 | 0.430 | 0.437 |
discussion|  0.175 | 0.189 | 0.227 | 0.248 | 0.297 | 0.326 | 0.334 | 0.363 |
directions| 0.124 | 0.139 | 0.176 | 0.191 | 0.203 | 0.206 | 0.212 | 0.226 |
greeting | 0.194 | 0.202 | 0.220 | 0.228 | 0.233 | 0.233 | 0.236 | 0.242 |
phoning|  0.149 | 0.163 | 0.202 | 0.211 | 0.229 | 0.244 | 0.245 | 0.273 |
posing | 0.345 | 0.352 | 0.426 | 0.480 | 0.570 | 0.590 | 0.613 | 0.674|
purchases |  0.212 | 0.226 | 0.243 | 0.253 | 0.270 | 0.272 | 0.275 | 0.285  |
sitting | 0.200 | 0.201 | 0.206 | 0.207 | 0.212 | 0.216 | 0.217 | 0.226 |
sittingdown |0.245 | 0.266 | 0.331 | 0.332 | 0.346 | 0.365 | 0.369 | 0.383|
takingphoto |0.228 | 0.256 | 0.283 | 0.297 | 0.312 | 0.323 | 0.325 | 0.359|
waiting |0.210 | 0.254 | 0.284 | 0.292 | 0.308 | 0.312 | 0.316 | 0.346|
walkingdog | 0.143 | 0.165 | 0.182 | 0.185 | 0.201 | 0.202 | 0.208 | 0.219|
walkingtogether |0.218 | 0.230 | 0.268 | 0.282 | 0.293 | 0.300 | 0.308 | 0.326 |
Average | 0.214 |	0.229	| 0.261	| 0.273 | 0.292 |	0.299	| 0.304 |	0.319| 0.274 | 

### Lie alone (MAE) :

H3.6M | 80ms | 160ms | 320ms | 400ms | 560ms | 640ms | 720ms | 1000ms 
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:  | :----:
walking | 0.503 | 0.556 | 0.846 | 0.894 |  0.979 |  1.039 | 1.112 | 1.143 |-
eating | 1.016  | 1.114 | 1.193 | 1.208 |  1.217 |  1.254 | 1.280 | 1.339 |-
smoking |  0.280 |  0.559 | 0.946 | 1.116 | 1.231 |1.375 | 1.410 | 1.565 |-
discussion| 0.352 | 0.483 | 0.852 | 1.073 | 1.349 | 1.552 | 1.628 |  1.714 |-
directions | 0.236  | 0.390 | 0.754 | 0.912  | 1.032  | 1.048 | 1.115 | 1.257 |-
greeting | 0.937 | 0.998 |  1.200 | 1.263 | 1.320 | 1.329  | 1.350  | 1.412 | -
phoning | 0.486 |  0.625 | 0.992 | 1.103 | 1.306 | 1.432 |  1.461 |  1.739  |-
posing |  0.445 | 0.522 | 1.269 | 1.798 |  1.911 |  2.115 | 2.327 |  2.435 |-
purchases | 1.111  |  1.268 | 1.433 | 1.539 | 1.699 | 1.740 | 1.774  | 1.858  |-
sitting |  1.003 | 1.026 | 1.057 | 1.093 | 1.117 | 1.161 | 1.181 | 1.248 |-
sittingdown | 0.461 | 0.666 | 1.286 | 1.302 | 1.465 | 1.657 | 1.682 | 1.824 |-
takingphoto | 1.281 | 1.564 | 1.851 | 1.978 | 2.126 | 2.253 | 2.275 | 2.582  |-
waiting | 1.074 | 1.527 |  1.745 |  1.811 | 1.985 | 2.008  |  2.158 | 2.262 |-
walkingdog | 0.450 | 0.670 | 0.828 | 0.837 | 1.025 | 1.030 | 1.090  | 1.170 |-
walkingtogether | 1.161 | 1.296 | 1.577 | 1.725 | 1.805 | 1.980 | 2.070 | 2.153 |-

### Kendall alone (MAE) :

H3.6M | 80ms | 160ms | 320ms | 400ms | 560ms | 640ms | 720ms | 1000ms  
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:  | :----:
walking | 0.465 | 0.542 | 0.790 | 0.833 | 0.897 | 0.994 | 1.015 | 1.120|  -
eating |1.026 | 1.117 | 1.178 | 1.195 | 1.223 | 1.243 | 1.257 | 1.350|-
smoking | 0.253 | 0.555 | 0.953 | 1.125 | 1.244 | 1.283 | 1.320 | 1.485 |-
discussion |0.571 | 0.686 | 0.849 | 0.954 | 1.309 | 1.548 | 1.690 | 1.723 |-
directions |0.212 | 0.344 | 0.780 | 0.909 | 0.997 | 1.032 | 1.122 | 1.251 |-
greeting |0.926 | 0.989 | 1.178 | 1.235 | 1.287 | 1.294 | 1.306 | 1.394 | -
phoning| 0.467 | 0.600 | 1.030 | 1.112 | 1.324 | 1.444 | 1.482 | 1.805 |-
posing |0.432 | 0.504 | 1.258 | 1.838 | 2.145 | 2.320| 2.408 | 2.531 |-
purchases | 1.110 | 1.281 | 1.488 | 1.568 | 1.735 | 1.794 | 1.817 | 1.885 |-
sitting | 1.022 | 1.041 | 1.107 | 1.115 | 1.173 | 1.184 | 1.230 | 1.281 |-
sittingdown |0.458 | 0.659 | 1.173 | 1.293 | 1.470 | 1.521 | 1.670 | 1.727|-
takingphoto | 1.324 | 1.606 | 1.869 | 1.974 | 2.168 | 2.252 | 2.304 | 2.608 |-
waiting | 1.072 | 1.391 | 1.532 | 1.670 | 1.846 | 1.979 | 2.134 | 2.217 |-
walkingdog | 0.438 | 0.630 | 0.752 | 0.826 | 1.016 | 1.055 | 1.077 | 1.165 |-
walkingtogether | 1.097 | 1.248 | 1.412 | 1.549 | 1.641 | 1.727 | 1.918 | 2.081 |-
  
### Euclidien Space :

H3.6M | 80ms | 160ms | 320ms | 400ms | 560ms | 640ms | 720ms | 1000ms  
:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:  | :----:


## Citation

If you find this useful, please cite our work as follows:

```
Not available.
```

## Notes
The repository is still under construction. Please let me know if you encounter any issues.

Best, 
