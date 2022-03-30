import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage.interpolation import rotate
import os
import scipy.io as sio

class HSIDataset:
    def __init__(self, data_name, pcaComponents, patchsz, train_val_test, random_state):
        super().__init__()
        info = DatasetInfo.info[data_name]
        data = sio.loadmat(info['data_path'])[info['data_key']]
        label = sio.loadmat(info['label_path'])[info['label_key']]

        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.int32)
        data = self.Normalize(data)
        data = self.applyPCA(data, pcaComponents)

        self.data = self.addMirror(data, patchsz)
        self.label = label
        self.x_patch, self.y_patch = self.createPatches(self.data, self.label, patchsz)
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.splitTrainValTest(
            self.x_patch, self.y_patch, train_val_test, random_state)


    def Normalize(self, x):
        h, w, c = x.shape
        new_x = x.reshape((-1, c))
        new_x -= np.min(new_x)
        new_x /= np.max(new_x)
        new_x = new_x.reshape((h, w, c))
        return new_x

    def applyPCA(self, x, n_components):
        h, w, c = x.shape
        new_x = np.reshape(x, (-1, c))
        pca = PCA(n_components=n_components, whiten=True)
        new_x = pca.fit_transform(new_x)
        new_x = np.reshape(new_x, (h, w, n_components))
        return new_x

    def padWithZeros(self, x, patchsz):
        dx = patchsz // 2
        h, w, c = x.shape
        new_x = np.zeros((h+2*dx, w+2*dx, c))
        new_x[dx:-dx, dx:-dx] = x
        return new_x

    def addMirror(self, x, patchsz):
        dx = patchsz // 2
        x = self.padWithZeros(x, patchsz)
        for i in range(dx):
            x[i, :, :] = x[2*dx-i, :, :]
            x[:, i, :] = x[:, 2*dx-i, :]
            x[-i-1, :, :] = x[-(2*dx-i)-1, :, :]
            x[:, -i-1, :] = x[:, -(2*dx-i)-1, :]
        return x

    def createPatches(self, data, label, patchsz):
        nonzero = np.nonzero(label)
        sample_ind = list(zip(*nonzero))
        num_sample = len(sample_ind)
        patched_data = np.zeros((num_sample, patchsz, patchsz, data.shape[2], 1), dtype=np.float32)
        patched_label = np.zeros(num_sample, dtype=np.int32)
        dx = patchsz // 2
        for i, (x, y) in enumerate(sample_ind):
            patched_data[i] = np.expand_dims(data[x:x+2*dx+1, y:y+2*dx+1], axis=-1)
            patched_label[i] = label[x, y] - 1
        return patched_data, patched_label

    def splitTrainValTest(self, x_patch, y_patch, train_val_test, random):
        x_train, x_val_test, y_train, y_val_test = train_test_split(x_patch, y_patch,
                                                                    train_size=train_val_test[0],
                                                                    random_state=random,
                                                                    stratify=y_patch)
        x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test,
                                                        test_size=train_val_test[2],
                                                        random_state=random,
                                                        stratify=y_val_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def augmentData(self, x_train, y_train):
        temp = np.copy(x_train)
        y_temp = np.copy(y_train)
        for i in range(temp.shape[0]):
            rand_ind = random.randint(0, 4)
            if rand_ind == 0:
                temp[i] = np.fliplr(temp[i])
            elif rand_ind == 1:
                temp[i] = np.flipud(temp[i])
            elif rand_ind == 2:
                temp[i] = rotate(temp[i], angle=90, reshape=False, prefilter=False)
            elif rand_ind == 3:
                temp[i] = rotate(temp[i], angle=180, reshape=False, prefilter=False)
            else:
                temp[i] = rotate(temp[i], angle=270, reshape=False, prefilter=False)
        x_train = np.concatenate((x_train, temp))
        y_train = np.concatenate((y_train, y_temp))
        return x_train, y_train

class DatasetInfo:
    info = {
        'indian':{
            'data_path':'./data/Indian/Indian_pines_corrected.mat',
            'label_path':'./data/Indian/Indian_pines_gt.mat',
            'data_key':'indian_pines_corrected',
            'label_key':'indian_pines_gt',
            'target_names': [
        'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers']
        },
        'ksc':{
            'data_path': './data/KSC/KSC.mat',
            'label_path': './data/KSC/KSC_gt.mat',
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'target_names':[
                'Scrub', 'Willow swamp', 'Cabbage palm hammock',
                'Cabbage palm/oak hammock', 'Slash pine', 'Oak/broadleaf hammock',
                'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
                'Cattail marsh', ' Salt marsh', ' Mud flats', 'Water'
            ]
        },
        'pavia':{
            'data_path':'./data/Pavia/Pavia.mat',
            'label_path':'./data/Pavia/Pavia_gt.mat',
            'data_key':'pavia',
            'label_key':'pavia_gt'
        },
        'salinas':{
            'data_path':'./data/Salinas/Salinas.mat',
            'label_path':'./data/Salinas/Salinas_gt.mat',
            'data_key':'salinas_corrected',
            'label_key':'salinas_gt',
            'target_names':[
                'Broccoli green weeds 1','Broccoli green weeds 22','Fallow',
                'Fallow rough plow','Fallow smooth','Stubble','Celery',
                'Grapes untrained','Soy vineyard develop','Corn senesced green weeds',
                'Lettuce romaine 4wk', ' Lettuce romaine 5wk','Lettuce romaine 6wk',
                ' Lettuce romaine 7wk', 'Vineyard untrained', 'Vineyard vertical trellis'
            ]
        }
    }




























