import os.path
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from model import HybridSN
from dataset import HSIDataset
import matplotlib.pyplot as plt


def train(data_name, n_components=30, patchsz=5, train_val_test=None, lr=1e-3, random_state=11413):
    if train_val_test is None:
        train_val_test = [0.2, 0.1, 0.7]
    HSI = HSIDataset(data_name, n_components, patchsz, train_val_test, random_state)
    x_train, y_train = HSI.x_train, HSI.y_train
    x_val, y_val = HSI.x_val, HSI.y_val
    input_shape = x_train[0].shape
    in_channel = x_train[0].shape[2]
    num_class = np.max(y_train) + 1
    model = HybridSN(nc=num_class)
    adam = Adam(learning_rate=lr, decay=1e-6)
    model.compile(optimizer=adam, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    model_path = './model_ckp/{}+npca{}+patchsz{}+lr{}+tvt{}-{}-{}'.format(data_name,n_components,
                                                                    patchsz, lr, train_val_test[0],
                                                                    train_val_test[1], train_val_test[2])
    if os.path.exists(model_path + '.index'):
        print('-----------------loading saved model---------------------')
        model.load_weights(model_path)
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[checkpointer],
                        shuffle=True)
    model.summary()
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train(data_name='indian', patchsz=25, n_components=30)