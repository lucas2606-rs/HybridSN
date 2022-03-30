from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dropout, Flatten, Dense, Conv3D, Reshape
import tensorflow as tf

class HybridSN(Model):
    def __init__(self, nc):
        super(HybridSN, self).__init__()
        self.c1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')
        self.c2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')
        self.c3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')
        self.c4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.flatten = Flatten()
        self.f1 = Dense(256, activation='relu')
        self.d1 = Dropout(0.4)
        self.f2 = Dense(128, activation='relu')
        self.d2 = Dropout(0.4)
        self.f3 = Dense(nc, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        shape = x.shape
        x = tf.reshape(x, [-1, shape[1], shape[2], shape[3]*shape[4]])
        x = self.c4(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y

