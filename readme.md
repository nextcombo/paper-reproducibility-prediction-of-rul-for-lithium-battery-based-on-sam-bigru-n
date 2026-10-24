# Paper Reproducibility! Prediction of RUL for Lithium Battery Based on SAM-BiGRU Network

Preface
This issue is based on the advanced life cycle engineering research center (CALCE) lithium battery experimental data from the University of Maryland, USA. It reproduces the model and comparative experiments proposed in the paper "Lithium Battery RUL Prediction Based on SAM-BiGRU Network". Through an in-depth analysis and experimental verification of this advanced deep learning model, we aim to provide a methodological path and data support for subsequent research on health management systems for lithium batteries (BHM), which has significant learning and reference value.
1 Reproduction Overview
1.1 Model Evaluation:
1.2 Predictive Visualization:
We also provide detailed materials and video explanations, including tutorials on parameter adjustments, replacement of prediction tasks, etc. The code is commented line by line with detailed parameter introductions:
● Dataset: Lithium Battery Life Test Public Dataset from CALCE
● Environment Framework: Python 3.9, PyTorch 2.1 or above versions can be run
● Usage Object: Those who need the requirements of the paper or graduation design
● Code Guarantee: The code is commented in detail, ready to run without modification.
Click the end of the article to read the original text and get the code!
2 Reproduction of the Paper Brief
2.1 Abstract
To address the issues of low accuracy in existing methods for predicting the remaining useful life (RUL) of lithium batteries, this paper proposes a bidirectional gated recurrent unit (BiGRU) network model based on self-attention mechanism (SAM). The capacity data of lithium batteries is used as the input sequence for this model. Through the SAM, key moments in the history of battery capacity are captured and assigned weights. The BiGRU model is then utilized to learn the capacity degradation trend, thereby achieving the prediction of remaining life. The proposed method was applied to the CS2 series of 35, 36, and 37 lithium batteries from the CALCE dataset. Experimental results show that the prediction errors for batteries 35 and 36 are within 1.5%, while for battery 37, the error is 2.22%.
2.2 Introduction to the Paper Model
The capacity degradation of lithium batteries is a sequential problem. This paper fully captures the characteristics of key time points in long-distance sequence data through self-attention mechanisms and assigns weights. It effectively acquires the historical and future information of long sequence data using BiGRU neural networks, which can form an effective complement. By combining self-attention mechanisms and BiGRU networks, an SAM-BiGRU model is constructed.
Firstly, the capacity sequence of lithium batteries x1~xt is linearly transformed and divided to serve as input for the self-attention mechanism. Subsequently, it is fed into the self-attention model where each component undergoes dot product operations before outputting. The output results are then concatenated to obtain the weight distribution of the input information. Finally, the attention output results are used as input to the BiGRU network, which utilizes the processing capability of the BiGRU network to predict the remaining capacity of lithium batteries based on time series data.
2.3 Introduction to the Paper Dataset
\t  
This article uses experimental data from lithium battery experiments conducted by the Advanced Life Cycle Engineering Research Center (CALCE) at the University of Maryland. The dataset consists of three batteries with serial numbers CS2_35, CS2_36, and CS2_37, specifically targeting LiCoO2 batteries. The lithium battery has a nominal capacity of 1.1 Ah, and its charging and discharging protocols follow standard constant current and constant voltage protocols. The charging process involves a constant current of 1 C until the voltage reaches 4.2 V, followed by constant voltage charging until the current drops to below 0.05 A. Discharging is done at a constant current of 1 C until the voltage reaches 2.7 V. All experiments were conducted under room temperature conditions (25°C).

When the test capacity of the battery falls below 70% of its nominal capacity, it can be considered as failed. In this study, the threshold for battery failure is set at 0.77 Ah.

3. Predictive Model Based on SAM-BiGRU
3.1 Data Importation
Taking the CS2_35 dataset as an example, we perform data preprocessing:
- Calculate local means and standard deviations using sliding windows to identify outliers and interpolate them.

3.2 Parameter Setting and Model Training
- 3.3 Model Evaluation and Comparison Experiments
Note: The original paper does not provide source code, and the training parameters are not fully detailed. We aim to reproduce the results as closely as possible, and any discrepancies are expected.
The recreated model and data processing workflow can serve as a reference and learning resource for beginners or those requiring papers.
Click to download: Full dataset, Python code
Previously interesting content:
Time series prediction: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed prediction (Part I): Introduction and preprocessing of datasets
Wind Speed Prediction (II) EMD-LSTM Model Based on Pytorch
Wind Speed Prediction (III) EMD-LSTM-Attention Model
Wind Speed Prediction (IV) EMD-Transformer Model Based on Pytorch
Wind Speed Prediction (5) EMD-CNN-LSTM Model Based on Pytorch
Wind Speed Prediction (VI) - EMD-CNN-GRU Parallel Model Based on Pytorch
CEEMDAN + Combination Forecasting Model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Forecasting Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Forecasting Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Forecasting Model (CNN-Transformer + ARIMA)
Multi-feature Variable Sequence Prediction (I) - CNN-LSTM Wind Speed Prediction Model
Multi-feature Variable Sequence Prediction (II) - CNN-LSTM-Attention Wind Speed Prediction Model
Multi-feature Variable Sequence Prediction (III) - CNN-Transformer Wind Speed Prediction Model
Multi-feature Variable Sequence Prediction (IV) - Transformer-BiLSTM Wind Speed Prediction Model
Multi-feature Variable Sequence Prediction (V) CEEMDAN+CNN-LSTM Wind Speed Prediction Model
Multi-feature Variable Sequence Prediction (VI) CEEMDAN+CNN-Transformer Wind Speed Prediction Model
Predictive Model Based on the Optimized Algorithm of Sparrows SSA-CEEMDAN-BiLSTM-Attention
CEEMDAN-Transformer-BiGRU Prediction Model Based on Optimized Algorithm SSA for Sparrows
Super Predictive Model: Secondary Decomposition-Combination Prediction
Multi-feature variable sequence prediction (7) CEEMDAN+Transformer-BiLSTM prediction model  
Multi-feature variable sequence prediction (8) CEEMDAN-SSA-BiLSTM prediction model based on the Plumage Optimization Algorithm  
Multi-feature variable sequence prediction (11) TCN-GRU prediction model based on Pytorch  
VMD + CEEMDAN secondary decomposition, CNN-LSTM prediction model  
VMD + CEEMDAN secondary decomposition, CNN-Transformer prediction model  
VMD + CEEMDAN secondary decomposition, Transformer-BiGRU prediction model  
Prediction model based on SSA of Plumage Optimization Algorithm - code family set  
 | Superior combination prediction model!
High innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention prediction model  
 | Superior combination prediction model!
Lowest price in the network | Family set continuously updated!  
| Based on TCN-SENet + BiGRU-GlobalAttention parallel prediction model
Secondary decomposition - innovative model prediction family set  
 | BiTCN-BiGRU-CrossAttention integrated spatiotemporal features high innovation prediction model  
Combination prediction update | Superior combination prediction model!  
Spatiotemporal feature integration BiTCN-Transformer parallel prediction model  
| Based on multi-level attention mechanism parallel prediction model  
 | CEEMDAN-CNN-GRU-GlobalAttention + XGBoost combination prediction  
Multi-step prediction series | LSTM, CNN, Transformer, TCN, serial, parallel model collection  
EN  
\t2. Friends who like can click to follow, give the article a "like", share the circle of friends or discuss groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
