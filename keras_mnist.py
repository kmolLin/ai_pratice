# # import the necessary packages
#
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import SGD
# from sklearn import datasets
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
# help="path to the output loss/accuracy plot")
# args = vars(ap.parse_args())
#
# # grab the MNIST dataset (if this is your first time running this
# # script, the download may take a minute -- the 55MB MNIST dataset
# # will be downloaded)
# print("[INFO] loading MNIST (full) dataset...")
# dataset = datasets.fetch_mldata("MNIST Original")
# # scale the raw pixel intensities to the range [0, 1.0], then
# # construct the training and testing splits
# data = dataset.data.astype("float") / 255.0
# (trainX, testX, trainY, testY) = train_test_split(data,
#                                                   dataset.target, test_size=0.25)
# # convert the labels from integers to vectors
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.transform(testY)
#
# # define the 784-256-128-10 architecture using Keras
# model = Sequential()
# model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
# model.add(Dense(128, activation="sigmoid"))
# model.add(Dense(10, activation="softmax"))
#
# # train the model using SGD
# print("[INFO] training network...")
# sgd = SGD(0.01)
# model.compile(loss="categorical_crossentropy", optimizer=sgd,
# metrics=["accuracy"])
# H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)
#
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=128)
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
#                             target_names=[str(x) for x in lb.classes_]))
#
# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig(args["output"])
#

# Numpy, tensorflow, Keras, matplotlib都是需要的
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# 現在如果不是localhost載入資料集，都應該會要透過https進行，
# 多了一道ssl認證，有時候會compile不過，下面這兩句可以讓他不用再進行ssl認證
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# loading...for training and testing 用，這邊大家設計的變數名稱都不一樣，
# 我習慣用X當作feature，lbl當作標籤。可以分成下面四種
(train_x, train_lbl), (test_x, test_lbl) = mnist.load_data()

# data先做簡單的預處理，先把它換成(28*28=784)個input。ps:也可以使用預設的flatten層來把維度降低成單維度啦
# 為了簡化神經層的處理，用numpy簡單處理一下
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)

# 之後會將各個像素降低到0.00~1.00之間，才可以套入神經網路的activation function
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255
test_x = test_x / 255

# one hot encoding，將label（手寫辨識結果）轉為1-d array
# 原本 label 應該是某數字 （e.g. 9 -->  [0,0,0,0,0,0,0,0,0,1])
# 一樣也是方便之後神經網路處理（最後輸出層的node數量為10個（0~9））
train_lbl = np_utils.to_categorical(train_lbl, 10)
test_lbl = np_utils.to_categorical(test_lbl, 10)

# 初始化一個model，用sequential宣告一個深度網路，keras + tensorflow就可以一層一層add起來
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

# 要設計更多層也是ＯＫ，但會變很慢...
# 取特徵通常用relu的激勵函數，而output判斷出結果（0~9）的時候通常會用softmax，這部分有一點數學...可以估狗ＸＤ
model.add(Dense(10, activation='softmax'))

# 優化器（讓電腦知道應該往那個方向提供使用者需要的結果）目前這組data是要達到降低loss（crossentropy...）
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit就是訓練已知的dataset，真的丟進去上面建立的神經網路跑。
# batch_size（一次會丟多少個instance）epochs（要跑幾輪/幾波）
# validation_data(不是必須，但可以用來驗證模型的準確度。
# ps:建立好的model會是fit train_x的結果。test_x是拿來判斷model好不好的（準？）
history = model.fit(train_x, train_lbl,
                    batch_size=128, epochs=8,
                    verbose=1, validation_data=(test_x, test_lbl))

# 預測label是什麼
prediction = model.predict_classes(test_x)

# 若有label的資料，可以evaluate評估testing的精準度（accuracy）...方法就是預測之後比對label
p = model.evaluate(test_x, test_lbl, verbose=2)

