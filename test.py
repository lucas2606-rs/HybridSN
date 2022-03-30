import os.path
from model import HybridSN
import scipy.io as sio
from dataset import HSIDataset, DatasetInfo
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def report(model, x_test, y_test, target_names):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    class_acc = classification_report(y_test, pred, target_names=target_names)
    confusion_mat = confusion_matrix(y_test, pred)
    score = model.evaluate(x_test, y_test, batch_size=32)
    test_loss = score[0]
    test_acc = score[1] * 100
    return class_acc, confusion_mat, test_loss, test_acc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = Normalized
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(data_name, n_components=30, patchsz=5, train_val_test=None, lr=1e-3, random_state=11413):
    if train_val_test is None:
        train_val_test = [0.2, 0.1, 0.7]
    model_path = './model_ckp/{}+npca{}+patchsz{}+lr{}+tvt{}-{}-{}'.format(data_name, n_components,
                                                                      patchsz, lr, train_val_test[0],
                                                                      train_val_test[1], train_val_test[2])
    HSI = HSIDataset(data_name=data_name, pcaComponents=n_components,
                     patchsz=patchsz, train_val_test=train_val_test, random_state=random_state)
    x_test, y_test = HSI.x_test, HSI.y_test
    info = DatasetInfo.info[data_name]
    target_names = info['target_names']
    num_class = np.max(y_test) + 1
    model = HybridSN(num_class)
    adam = Adam(learning_rate=lr, decay=1e-6)
    model.compile(optimizer=adam, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    if os.path.exists(model_path + '.index'):
        print('-----------------loading trained model---------------------')
        model.load_weights(model_path)
    class_acc, cm, test_loss, test_acc = report(model, x_test, y_test, target_names)
    class_acc = str(class_acc)
    cm_str = str(cm)
    print('Test loss:{}'.format(test_loss))
    print('Test acc:{}%'.format(test_acc))
    print('Classification result:')
    print(class_acc)
    print('Confusion matrix:')
    print(cm_str)

    report_save_path = './result/{}+npca{}+patchsz{}+tvt{}-{}-{}'.format(data_name, n_components,
                                                                      patchsz, train_val_test[0],
                                                                      train_val_test[1], train_val_test[2])
    if not os.path.exists(report_save_path):
        os.makedirs(report_save_path)
    file_name = os.path.join(report_save_path, 'report.txt')
    with open(file_name, 'w') as f:
        f.write('Test loss:{}'.format(test_loss))
        f.write('\n')
        f.write('Test acc:{}%'.format(test_acc))
        f.write('\n')
        f.write('\n')
        f.write('Classification result:\n')
        f.write('{}'.format(class_acc))
        f.write('\n')
        f.write('Confusion matrix:\n')
        f.write('{}'.format(cm_str))
        f.write('\n')
    print('-------------successfully create report.txt!-------------------')

    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cm, classes=target_names, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(report_save_path, 'confusion_mat_without_norm.png'))
    print('------------succesfully generate confusion matrix pic!-----------')


if __name__ == '__main__':
    test('indian', patchsz=25, n_components=30)