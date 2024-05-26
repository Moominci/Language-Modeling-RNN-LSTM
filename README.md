# Language-Modeling
Language-Modeling using RNN, LSTM models trained by "shakespeare_train.txt"
generating characters with our trained model.

# Assignment submitted by MinSeok Yoon
- Seoul National University of Science and Technology, Korea
- Department of Data Science
- Data-Driven User Analysis Lab
#### [Data-Driven User Analysis Lab] (https://ddua.seoultech.ac.kr/index.do)

# Assignment Reports
## Model Architecture Overview

### RNN Model 1 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| RNN-1           | [-1, 30, 128] | 20,480  |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 28,865  
**Trainable params:** 28,865  
**Non-trainable params:** 0  

| Metric                        | Size (MB) |
|-------------------------------|-----------|
| Input size                    | 0.00      |
| Forward/backward pass size    | 0.01      |
| Params size                   | 0.11      |
| **Estimated Total Size (MB)** | **0.12**  |

### RNN Model 2 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| RNN-1           | [-1, 30, 256] | 33,280  |
| Linear-2        | [-1, 100, 65] | 16,705  |

**Total params:** 49,985  
**Trainable params:** 49,985  
**Non-trainable params:** 0  

| Metric                        | Size (MB) |
|-------------------------------|-----------|
| Input size                    | 0.00      |
| Forward/backward pass size    | 0.02      |
| Params size                   | 0.19      |
| **Estimated Total Size (MB)** | **0.21**  |


### LSTM Model 1 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| LSTM-1          | [-1, 30, 128] | 99,840  |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 108,225  
**Trainable params:** 108,225  
**Non-trainable params:** 0  

| Metric                        | Size (MB) |
|-------------------------------|-----------|
| Input size                    | 0.00      |
| Forward/backward pass size    | 0.01      |
| Params size                   | 0.41      |
| **Estimated Total Size (MB)** | **0.43**  |

### LSTM Model 2 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| LSTM-1          | [-1, 30, 256] | 263,168 |
| Linear-2        | [-1, 100, 65] | 16,705  |

**Total params:** 279,873  
**Trainable params:** 279,873  
**Non-trainable params:** 0  

| Metric                        | Size (MB) |
|-------------------------------|-----------|
| Input size                    | 0.00      |
| Forward/backward pass size    | 0.02      |
| Params size                   | 1.07      |
| **Estimated Total Size (MB)** | **1.09**  |






## Softmax Function
- Softmax function with a temperature parameter T can be written as: 
y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}  
