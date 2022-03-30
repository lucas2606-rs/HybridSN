import spectral
import matplotlib.pyplot as plt
from dataset import HSIDataset
import numpy as np
from model import HybridSN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os

def generate_pred_map(data_name, n_components=30, patchsz=5, train_val_test=None, lr=1e-3, random_state=11413):
    if train_val_test is None:
        train_val_test = [0.2, 0.1, 0.7]
    HSI = HSIDataset(data_name=data_name, pcaComponents=n_components,
                     patchsz=patchsz, train_val_test=train_val_test, random_state=random_state)
    x_patch, y_patch = HSI.x_patch, HSI.y_patch
    label = HSI.label
    nonzero = np.nonzero(label)
    sample_ind = list(zip(*nonzero))
    num_sample = len(sample_ind)
    pred_map = np.zeros_like(label)
    input_shape = x_patch[0].shape
    in_channel = x_patch[0].shape[2]
    num_class = np.max(label)
    model_path = './model_ckp/{}+npca{}+patchsz{}+lr{}+tvt{}-{}-{}'.format(data_name, n_components,
                                                                      patchsz, lr, train_val_test[0],
                                                                      train_val_test[1], train_val_test[2])
    model = HybridSN(nc=num_class)
    adam = Adam(learning_rate=lr, decay=1e-6)
    model.compile(optimizer=adam, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    if os.path.exists(model_path + '.index'):
        print('-----------------loading trained model---------------------')
        model.load_weights(model_path)
    pred = model.predict(x_patch, batch_size=32)
    pred = np.argmax(pred, axis=1)
    for i, (x, y) in enumerate(sample_ind):
        pred_map[x, y] = pred[i] + 1
    pred_pic_savepath = './result/{}+npca{}+patchsz{}+tvt{}-{}-{}'.format(data_name, n_components,
                                                                      patchsz, train_val_test[0],
                                                                      train_val_test[1], train_val_test[2])
    if not os.path.exists(pred_pic_savepath):
        os.makedirs(pred_pic_savepath)
    predict_image = spectral.imshow(classes=pred_map.astype(int), figsize=(10, 10))
    plt.savefig(os.path.join(pred_pic_savepath, 'pred_map.jpg'))
    print('------------------sucessfully saved pred map!----------------------')

generate_pred_map('indian', patchsz=25, n_components=30)