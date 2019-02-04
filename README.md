# stock_price_prediction
The project predicts the real Google stock price by implementing LSTMs (Long Short Term Memory RNNs).  
### Python packages
absl-py             0.7.0  
astor               0.7.1  
cycler              0.10.0  
gast                0.2.2  
grpcio              1.18.0  
h5py                2.9.0  
Keras               2.2.4  
Keras-Applications  1.0.7  
Keras-Preprocessing 1.0.8  
kiwisolver          1.0.1  
Markdown            3.0.1  
matplotlib          3.0.2  
numpy               1.16.1  
pandas              0.24.0  
Pillow              5.4.1  
pip                 19.0.1  
protobuf            3.6.1  
pyparsing           2.3.1  
python-dateutil     2.7.5  
pytz                2018.9  
PyYAML              3.13  
scikit-learn        0.20.2  
scipy               1.2.0  
setuptools          40.7.2  
six                 1.12.0  
sklearn             0.0  
tensorboard         1.12.2  
tensorflow          1.12.0  
termcolor           1.1.0  
virtualenv          16.3.0  
Werkzeug            0.14.1  
wheel               0.32.3  
### Goal
Train the RNN by using the stock price from 2012 to 2016 to predict the stock price of Jan 2017  
### First try
Code: rnn.py  
![](/result/first_prediction.png)  
Scaled mean squared error: 0.32343  
### Ways to optimize
1. Choose larger timesteps, change 60 to 120. This means predicting the current price by using the previous 120 days of data instead of 60 days  
2. Add two more indicators (highest price, lowest price) to train the model  
3. Add two more hidden layers (each hidden layer contains one LSTM layer and one Dropout layer)  
4. Do parameter tuning to find better values for variables of batch_size, epochs, and output_units  
### Second try
Code: rnn_optimized.py  
![](/result/improved_prediction.png)  
Scaled mean squared error: 0.32273  
