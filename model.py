import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

iden = 'EURPLN=x'
df = yf.download(iden, start='2020-03-21', end = '2023-03-21')

price = df[["Close"]].values.astype('float32')

#split the data
train_size = int(len(price) * 0.67)
test_size = len(price) - train_size

train, test = price[:train_size], price[train_size:]

def create_preds(dataset, w): #w = lookback period
    X = np.array([dataset[i:i+w] for i in range(len(dataset)-w)])
    y = np.array([dataset[i+1:i+w+1] for i in range(len(dataset)-w)])
    
    return torch.tensor(X), torch.tensor(y)

#create prediction dataset (tensors)
lookback = 10
X_train, y_train = create_preds(train, w = lookback)
X_test, y_test = create_preds(test, w = lookback)

#create network
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 50,
                            num_layers = 1, batch_first = True)
        self.linear = nn.Linear(50,1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(f'Using {device}')

#initialize model
model = AirModel().to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

#load data
train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle= True, batch_size = 16)
test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle= False, batch_size = 16)
    
#train dataset
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    losses = []

    for X, y in train_loader:
        X = X.to(device)
        y= y.to(device)
    
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train).cpu())
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test).cpu())

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


#plotting data
with torch.no_grad():
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test2 = X_test.to(device), y_test.to(device)
    
    # shift train predictions for plotting
    train_plot = np.ones_like(price) * np.nan
    train_plot[lookback-1:train_size-1] = (model(X_train)).cpu()[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(price) * np.nan
    test_plot[train_size+lookback-1:len(price)-1] = (model(X_test)).cpu()[:, -1, :]        

style.use('bmh')
plt.plot(price, label = 'Actual price')
plt.title(f'{iden} stock price prediction')
plt.plot(train_plot, c='r', label = 'Price prediction on training data')
plt.plot(test_plot, c='g', label = 'Price prediction on testing data')
plt.legend()
plt.show()