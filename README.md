# Language-Modeling
Language-Modeling using RNN, LSTM models trained by "shakespeare_train.txt"
generating characters with our trained model.

# Assignment submitted by MinSeok Yoon
- Seoul National University of Science and Technology, Korea
- Department of Data Science
- Data-Driven User Analysis Lab
#### [Data-Driven User Analysis Lab] (https://ddua.seoultech.ac.kr/index.do)

# Assignment Reports
## **To run python files, please place the models in the "Models" folder in the same folder as the python file.
## ** The difference between ~~.py and ~~_drop.py is the difference between using a regular model and one with Dropout.
## Model Architecture Overview
- We used 2 vanilla RNN models and two LSTM models to compare model performance and conduct comparative experiments to see if applying performance enhancement techniques really worked.
- The structural differences between the first RNN, LSTM model and the second RNN, LSTM model are as follows.
  - num_layers : 2 => 3
  - Add Dropout Layer(=0.5) to prevent overfitting
    
### RNN Model 1 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| RNN-1           | [-1, 30, 128] | 20,480  |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 28,865  
**Trainable params:** 28,865  
**Non-trainable params:** 0  

### RNN Model 2 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| RNN-1           | [-1, 30, 128] | 25,344  |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 33,729  
**Trainable params:** 33,729  
**Non-trainable params:** 0  

### LSTM Model 1 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| LSTM-1          | [-1, 30, 128] | 99,840  |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 108,225  
**Trainable params:** 108,225  
**Non-trainable params:** 0  

### LSTM Model 2 Summary

| Layer (type)    | Output Shape  | Param # |
|-----------------|---------------|---------|
| LSTM-1          | [-1, 30, 128] | 131,584 |
| Linear-2        | [-1, 100, 65] | 8,385   |

**Total params:** 139,969  
**Trainable params:** 139,969  
**Non-trainable params:** 0 


## Experiments
### Hyperparameter
- Epochs = 10
- Batch size : 64
- hidden_size = 128
- Optimizer : AdamW
- Learning rate : 0.002
- Criterion = Cross Entropy Loss

### Average loss plot
- Average loss plot the average loss values for training and validation comparing RNN and LSTM
### Model RNN 1 & LSTM 1
<img src="./images/loss_plot_new.jpg" width="680" height="450" alt="Model 1 Average loss plot">

- Analysis Model Performance
  - The RNN & LSTM model's training loss decreases steadily over the epochs, indicating that the model is learning from the training data.
  - The validation loss also decreases initially but The LSTM model's validation loss decreases more significantly and consistently compared to the RNN model, and it achieves lower validation loss values overall.


### Model RNN 2 & LSTM 2
<img src="./images/loss_plot_drop.jpg" width="680" height="450" alt="Model 2 Average loss plot">

- Analysis Model Performance
  - Certainly, we see that attempts to improve the model performance (dropout, more layers) have reduced the loss of model learning.
  - We find that we start with a lower loss than in previous experiments (Model 1), and that we can obtain a lower loss.
  - Train Loss and Valid Loss for each model show a pattern of trying to represent the same value.

### Generation Performance Analysis Report
- We need to provide at least 100 pieces of length of 5 different samples generated based on different seed characters.
- We experimented with generating 200 words to see and evaluate more generation results.

### Analysis of the Role and Impact of the Temperature Parameter 'T'

#### Formula
The temperature parameter 'T' is used in the softmax function as follows:

\[ y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}} \]

#### Role of Temperature
The temperature parameter 'T' is used to control the uncertainty during the sampling process of the model's output. Depending on the value of 'T', the randomness in the sampling process changes, significantly affecting the diversity and consistency of the generated text.

1. **High Temperature (T > 1)**
    - **Increased Randomness**: A high 'T' value flattens the probability distribution, giving relatively high probabilities to less certain options. This increases the diversity of the generated text but can result in semantically inconsistent outputs.
    - **Example**: When 'T' is set to 2, the generated text can be highly diverse but may include grammatical errors or nonsensical sentences.
    - **RNN Example**: BUCKINGHAMENCERE: na,--fulthip ands! A oppuever: obeen! all. With that!--The noblos; Pain litted of Mardtim.'
    - **LSTM Example**: BUCKINGHAM: True? it so't,Where earth, kell doer griand;I will dof, ungeasy, as yiedHonder nned the swrackme,'En lougmment, thus yo't?

2. **Low Temperature (0 < T < 1)**
    - **Decreased Randomness**: A low 'T' value sharpens the probability distribution, concentrating high probabilities on the most certain options. This increases the consistency of the generated text but reduces diversity.
    - **Example**: When 'T' is set to 0.5, the generated text is more grammatically correct and consistent but may appear somewhat monotonous.
    - **RNN Example**: BUCKINGHAM: What should not of the fair face the hands, he will not that say the senaten day were be a bear his words And cloubs, with him the state the matter to make the one me that thy lives
    - **LSTM Example**: BUCKINGHAM: And, in good time, here by meine and the devil in thy peril, Her comfort of that world thing is merce that had eleven for you, lay of moon

3. **Temperature = 1**
    - **Baseline**: When 'T' is 1, the softmax function uses the original logit outputs to calculate probabilities. This results in sampling that follows the default probability distribution.
    - **Example**: When 'T' is 1, the generated text reflects the learned patterns of the model adequately, balancing randomness and consistency.
    - **RNN Example**: BUCKINGHAM: MARCIUS: Cate well. Being state, if they was the tribunes, I as well recoot'st to spetuce knows, The uther eye you, have but the earth The witnes, 
    - **LSTM Example**: BUCKINGHAM:Well, and is cereating to hers! Henr set shall pleased as we king That having abhold to bright hate of these breath What scorn to't to come is enhor

#### Impact of Temperature on Generated Results
Adjusting the temperature parameter 'T' yields different outcomes:

1. **Diversity**
    - A high 'T' value increases the likelihood of generating diverse words and sentences. This is useful for experimental or creative text generation.
    - A low 'T' value results in more predictable and consistent text. This is beneficial for practical or precise text generation.

2. **Consistency**
    - A high 'T' value can reduce the consistency of sentences, often leading to illogical or grammatically incorrect sentences.
    - A low 'T' value enhances sentence consistency, producing more natural text.

### Conclusion
The temperature parameter 'T' greatly influences the quality of the generated text. A high 'T' value produces more creative and diverse text, while a low 'T' value generates more consistent and logical text. Therefore, users should choose an appropriate 'T' value based on the purpose of the generated text. Experimenting with different 'T' values is essential to find the optimal setting for achieving natural and satisfying results.
