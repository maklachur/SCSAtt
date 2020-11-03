# SCSAtt Tracker
Efficient visual tracking with stacked channel-spatial attention learning (SCSAtt), IEEE Access 2020.
[Paper link](https://ieeexplore.ieee.org/document/9102303/)


<p align="center">
  <img src="motorRolling-SCSAtt.gif" />
</p>

## Summary
Siamese stacked channel-spatial attention learning for visual tracking where channel attention emphasizes 'what' informative part of the target image has to focus and spatial attention responsible for 'where' the informative part is located. Therefore, combining these two attention modules learn 'what' and 'where' to focus or suppress the target information to locate it efficiently.

![example](https://github.com/maklachur/SCSAtt/blob/master/Framework.jpg)

This work implemented using python with the PyTorch deep learning framework and performed all experiments on a desktop with Intel(R) Core(TM) i7-8700 CPU @ 3.20 GHz and Nvidia GeForce RTX 2080 Super GPU.

### To run this code, please consider the following instructions:
N.B. To re-implement this work, it is recommended to install python 3.6 with the PyTorch >= 1.0.

## Training
First, we need to configure the training dataset path from `config.py` file to start training the network.
After completing the training dataset configuratiion, run the `train.py` file. 

## Testing
To evaluate the model performance, configure testing benchmark path from `config.py` file and keep the pre-trained model in the "model" folder. We will use the pre-trained model during testing.
Finally, run the `test.py` file to get the success and precision plots.

N.B. To evaluate the other popoular benchmarks, configure the "experiments" variable from `test.py` file accordingly:
Example: benchmark_path is `data/OTB`, `data/UAV123`, `data/TC128`
        
        ExperimentOTB('benchmark_path', version=2015), #OR version='tb100'
        ExperimentOTB('benchmark_path', version='tb50'),
        ExperimentVOT('benchmark_path', version=2016),
        ExperimentVOT('benchmark_path', version=2017),
        ExperimentVOT('benchmark_path', version=2018),
        ExperimentTColor128('benchmark_path'),
        ExperimentUAV123('benchmark_path', version='UAV123'),
        ExperimentUAV123('benchmark_path', version='UAV20L'),
        ExperimentGOT10k('benchmark_path', subset='test'),
        ExperimentDTB70('benchmark_path'),
        ExperimentNfS('benchmark_path', fps=30),
        ExperimentNfS('benchmark_path', fps=240)


## Performance evaluation results
All of the results are computed and compared based on the official OTB and VOT toolkit ([OTB (50/100)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2016~2018)](http://votchallenge.net) 

#### OTB2015 / TColor128 / UAV123

| Dataset       | Success Score    | Precision Score  |
|:-------------:|:----------------:|:----------------:|
| OTB100        | 0.641            | 0.855            |
| OTB50         | 0.602            | 0.822            |
| TC128         | 0.549            | 0.744            |
| UAV123        | 0.547            | 0.776            |

### OTB50
![example](https://github.com/maklachur/SCSAtt/blob/master/otb50_result.jpg)
### OTB100
![example](https://github.com/maklachur/SCSAtt/blob/master/otb100_result.jpg)

### Dependencies/ Prerequisite

Install GOT-10k toolkit using the instructions from original site(recommended)before running this code:
[GOT-10K Toolkit installation guide](https://github.com/got-10k/toolkit#installation)

OR follow the simple steps to install:
#### Install the toolkit using pip (recommended):
```
1. pip install --upgrade got10k
```
#### Stay up-to-date:

```
2. pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```
If you  find useful please cite as,
```
@ARTICLE{9102303,  author={M. M. {Rahman} and M. {Fiaz} and S. K. {Jung}},  
journal={IEEE Access},   
title={Efficient Visual Tracking With Stacked Channel-Spatial Attention Learning},   
year={2020},  
volume={8}, 
pages={100857-100869}
}
```
