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

RNN Model Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            RNN-1              [-1, 30, 128]           20,480
            Linear-2           [-1, 100, 65]           8,385
================================================================
Total params: 28,865
Trainable params: 28,865
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.11
Estimated Total Size (MB): 0.12
----------------------------------------------------------------





## Softmax Function
- Softmax function with a temperature parameter T can be written as: 
y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}  
