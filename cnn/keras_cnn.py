from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten
from keras.callbacks import TensorBoard

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# check the value shape
print(f"x_train.shape : {x_train.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {x_test.shape}")
print(f"y_test.shape : {y_test.shape}")

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# class label 1-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(filters=32,
                 input_shape=(32, 32, 3),
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu')
          )

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu')
          )

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu')
          )

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu')
          )
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

print(model.output_shape)

model.add(Flatten())

print(f"output flatten layer shape: {model.output_shape}")

model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
tsb = TensorBoard(log_dir=".\\logs")
history_model1 = model.fit(x_train,
                           y_train,
                           batch_size=32,
                           epochs=20,
                           validation_split=0.2,
                           callbacks=[tsb])

model.save("my_first_model.h5")