import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Obtendo os dados da Yahoo Finance
ticker = "BTC-USD"  # Símbolo da ação a ser obtida
data = yf.download(ticker, start="2010-01-01", end="2024-06-30")  # Ajuste a data final para uma data futura

# Pré-processamento dos dados
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

# Função para dividir os dados
def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []
    
    # create all possible sequences of length lookback
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    
    x_train = data[:, :-1, :]
    y_train = data[:, -1, :]
    
    return [x_train, y_train]

lookback = 20  # choose sequence length
x_train, y_train = split_data(price, lookback)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

input_dim = 1
hidden_dim = 64
num_layers = 8
output_dim = 1
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Gerar previsões futuras
future_days = 10  # Número de dias futuros para prever
last_sequence = x_train[-1, :, :].unsqueeze(0)

model.eval()
future_data = []

for _ in range(future_days):
    with torch.no_grad():
        prediction = model(last_sequence)
        future_data.append(prediction.item())
        last_sequence = torch.cat((last_sequence[:, 1:, :], prediction.unsqueeze(0)), dim=1)

# Inverter a escala dos dados
future_data = np.array(future_data).reshape(-1, 1)
future_data = scaler.inverse_transform(future_data)

# Gerar datas futuras
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]

# Inverter a escala dos dados de treinamento
y_train_true = scaler.inverse_transform(y_train_lstm.detach().numpy())

# Plot dos resultados
train_dates = data.index[lookback:lookback+len(y_train_true)]
test_dates = future_dates

plt.figure(figsize=(12, 6))
plt.plot(train_dates, y_train_true, label='Historical Price')
plt.plot(test_dates, future_data, label='Future Predictions', color='r')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Formato das datas no eixo x
plt.legend()
plt.tight_layout()
plt.show()

