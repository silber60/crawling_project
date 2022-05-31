import numpy as np
import matplotlib.pyplot as plt
from tensorflow import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './models/news_data_winner_max_13_wordsize_9203.npy',
    allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


model = Sequential()
model.add(Embedding(9203, 300, input_length=13))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy',
              optimizer=opt, metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=15,
                     epochs=5, validation_data=(X_test, Y_test))



model.save('./models/wadiz_classfication_model_winner_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()