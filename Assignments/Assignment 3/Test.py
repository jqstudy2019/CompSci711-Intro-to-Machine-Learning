from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Dense

from matplotlib import pyplot as plt 

# cpu = fetch_openml(data_id=562)

# x_train, x_val, y_train, y_val = train_test_split(cpu.data, cpu.target, test_size=0.2, random_state=0)

# nn = Sequential()
# nn.add(Input((12,)))
# nn.add(Dense(30,activation="sigmoid"))
# nn.add(Dense(10,activation="sigmoid"))

# nn.add(Dense(1))

# nn.compile(optimizer="adam", loss="mse", metrics=["mse"])

# history = nn.fit(x_train, y_train, epochs=200,validation_data=(x_val,y_val),verbose=1)


# plt.plot(history.history["mse"], label="Training")
# plt.plot(history.history["val_mse"], label="Validation")
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.ylim(0,100)
# plt.legend()
# plt.show()

# print(min(history.history["val_mse"]))

# nn.evaluate(x_val,y_val)

# y_pred = nn.predict(x_val)
# print(y_pred)


dia = fetch_openml(data_id=44)

enc = OneHotEncoder(sparse_output=False)
tmp=[[x] for x in dia.target]
ohe_target=enc.fit_transform(tmp)

x_train, x_val, y_train, y_val = train_test_split(dia.data,
                                                    ohe_target, test_size=0.2,  
                                                    random_state=0)

nn = Sequential()
nn.add(Input((57,)))
nn.add(Dense(20,activation="relu"))
nn.add(Dense(10,activation="relu"))
nn.add(Dense(2,activation="softmax"))
nn.compile(optimizer="adam",
            loss="categorical_crossentropy", metrics=["accuracy"])

history = nn.fit(x_train, y_train, epochs=200,
                    validation_data=(x_val, y_val),verbose=1)

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,100)
plt.legend()
plt.show()

print(min(history.history["val_accuracy"]))

nn.evaluate(x_val,y_val)

y_pred = nn.predict(x_val)
print(y_pred)

print("Max Accuracy: ",max(history.history["val_accuracy"]) )