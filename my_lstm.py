import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from dataGen import *
import gc
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

# Velocity Array Data Pipeline
frq = np.arange(1, 9.5, 0.5)
amp = np.arange(0.01, 0.095, 0.005)
n_features = 1
n_removal = 1
y_hat = []


data, label = data_preparation(frq[:-2], amp[:-2])
val_data, val_label = data_preparation(frq[-2:], amp[-2:])


model = Sequential()
model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_features, data.shape[2])))
model.add(LSTM(250, activation='relu', return_sequences=True, input_shape=(n_features, data.shape[2])))
model.add(Dropout(0.25))
model.add(LSTM(250, activation='relu', return_sequences=True, input_shape=(n_features, data.shape[2])))
model.add(Dropout(0.25))
model.add(LSTM(250, activation='relu', return_sequences=True, input_shape=(n_features, data.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(200, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.fit(data, label, epochs=30, validation_data=(val_data, val_label), verbose=2, batch_size=20)

model.save('lstm.hdf5')

x_test = data[3, :, :]
x_test = x_test.reshape((1, 1, 200))

x_val = label[3, :, :]
xax = np.arange(data.shape[2])

y_hat = model.predict(x_test, verbose=1)
x_val = x_val.reshape(200)
y_hat = y_hat.reshape(200)

plt.plot(xax, x_val, label='sample data')
plt.plot(xax, y_hat, label='prediction')
plt.savefig('lstm_png.png')
gc.collect()

# animation setting
y1 = model.predict(data, verbose=1)
fig = plt.figure()
ax = plt.axes(xlim=(0, 150), ylim=(0, 1.1))
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.arange(data.shape[2])
    y = y1[i, :, :]
    line.set_data(x, y)
    return line,


anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
anim.save('lstm_mp4.mp4')
anim.save('lstm_gif.gif', writer='imagemagick')

