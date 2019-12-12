from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import cv2
from pprint import pprint

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.
    # cv2.imshow("test", x_test[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow(y_test[0])
    model = load_model("my_first_model.h5")
    print(model.summary())
    test_image = np.expand_dims(x_test[0], axis=0)

    ypre = model.predict(test_image)
    print(f"output is {ypre}")
