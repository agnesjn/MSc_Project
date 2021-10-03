import os
os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
from scipy.stats import wasserstein_distance


def normalise(x):
    return (x - x.min(1)[:, np.newaxis]) / (x.max(1)[:, np.newaxis]-x.min(1)[:, np.newaxis])

def prepCNNdata(wf0, wf1, nchan, sample_size, wfsize, wf_shift_from_0, split_train_test):
    wf_a = wf0.astype(np.float16)
    wf_n = wf1.astype(np.float16)
 
    wf_asum = np.sum(wf_a, axis=1)[..., :wfsize]
    wf_nsum = np.sum(wf_n, axis=1)[..., :wfsize]

    
    signals_wf = np.concatenate((wf_nsum, wf_asum))
 
    labels = np.concatenate((np.ones(len(wf_nsum)), np.zeros(len(wf_asum))))
    # labels = labels[:, None]
    perm = np.arange(signals_wf.shape[0])
    np.random.shuffle(perm)
    labels = labels[perm]
    signals_wf = signals_wf[perm]

    # Normalise data
    signals_wf = normalise(signals_wf)

    x_train = signals_wf[:int(len(signals_wf)*split_train_test)]
    y_train = labels[:int(len(signals_wf)*split_train_test)]
    x_test = signals_wf[int(len(signals_wf)*split_train_test):]
    y_test = labels[int(len(signals_wf)*split_train_test):]

    return x_train, y_train, x_test, y_test


def gene_dataset():
    AmBe_data = np.load("/home/featurize/data/biponator_dev/outputNS_Neutron_AmBe_Global_Source_Stack.npz")
    Po_data = np.load("/home/featurize/data/biponator_dev/outputNS_BiPo.npz")
    print(Po_data['wfs'].shape,AmBe_data['wfs'].shape)

    size = 30720 
    wfsize = 3072 # avoid obvious differences between neutron and alpha waveforms
    shift = 0
    perc = 0.66 # percentage of data to be used as training data
    xtrain, ytrain, xtest, ytest = prepCNNdata(Po_data['wfs'], AmBe_data['wfs'], 4, size, wfsize, shift, perc)
    np.save('xtrain', xtrain)
    np.save('ytrain', ytrain)
    np.save('xtest', xtest)
    np.save('ytest', ytest)


def gene_var_map():
    xtrain = np.load('xtrain.npy')
    ytrain = np.load('ytrain.npy')
    x1 = xtrain[ytrain == 0]
    x2 = xtrain[ytrain == 1]
    var_map = np.zeros((xtrain.shape[1]))
    for i in range(xtrain.shape[1]):
        print(i)
        var_map[i] = wasserstein_distance(x1[:, i], x2[:, i])

    np.save('var_map', var_map)


if __name__ == '__main__':
    gene_dataset()
    gene_var_map()
    