# LSTM-RNN
LSTM-RNN for Data Generation and Transfer Learning in Python.

# Introduction
The project presents a Keras implementation of LSTM-RNN for Synthetic Data Generation and Transfer Learning in Python. An RNN consisting of
multiple LSTM cells is suitable for evaluation of such a time series. Based on this quantization, the RNN treats the feature-set as a time 
series data with each repetition of a gesture being a time sample. The output isconverted to a vector by virtue of one-hot encoding in order for
it to be compatible with the LSTM cells. 

# Usage 
Clone or Download the repository and save the files in your Python directory. Pass your dataset as the input to the 'LSTM_data_gen.py' file
and then run the code.

# Results
<img src="Results\master_cost.fig" width="100" height="100"/><img src="Results\slave_cost.fig" width="100" height="100"/><img src="Results\standalone_cost.fig" width="100" height="100"/>
<img src="Results\master_cost.val" width="100" height="100"/><img src="Results\slave_val.fig" width="100" height="100"/><img src="Results\standalone_val.fig" width="100" height="100"/>

