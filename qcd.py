import numpy as np

signal = ['DY', 'WW', 'WZ', 'ZZ', 'Wjets', 's-channel', 't-channel', 'tW-channel', 'ttbar']
background = ['QCD_data']


def create_datasets3():
    test_frac = 0.3

    features = None
    n_signal = n_background = 0


    for s in signal:
        data = np.genfromtxt(s + '.csv', delimiter=',', skip_header=1)

        if features is None:
            features = np.empty((0, data.shape[1]))

        n_signal += data.shape[0]
        features  = np.concatenate([features, data])

    labels = np.ones((n_signal, 1))

    for b in background:
        data = np.genfromtxt(b + '.csv', delimiter=',', skip_header=1)

        if features is None:
            features = np.empty((0, data.shape[1]))

        n_background += data.shape[0]
        features  = np.concatenate([features, data])

    labels = np.concatenate([labels , np.zeros((n_background , 1))])


    perm = np.arange(features.shape[0])
    np.random.shuffle(perm)
    features = features[perm]
    labels   = labels[perm]

    np.save('QCD_features.npy', features)
    np.save('QCD_labels.npy', labels)

    i = int(len(features) * test_frac)
    features_test = features[:i]
    features_train = features[i:]
    labels_test = labels[:i]
    labels_train = labels[i:]

    np.save('QCD_features_train.npy', features_train)
    np.save('QCD_features_test.npy', features_test)
    np.save('QCD_labels_train.npy', labels_train)
    np.save('QCD_labels_test.npy', labels_test)

    return features_train, labels_train, features_test, labels_test


def load_datasets3():
    return np.load('QCD_features_train.npy'), np.load('QCD_labels_train.npy'), np.load('QCD_features_test.npy'), np.load('QCD_labels_test.npy')



#create_train_data()
#train_signal, train_background, test_signal, test_background = create_datasets()
train_signal, train_background, test_signal, test_background = load_datasets()


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


model = Sequential()
model.add(Dense(100, kernel_initializer='lecun_uniform', activation='relu', input_dim=11))
model.add(Dense(100, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='lecun_uniform', activation='relu'))


model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))
sgd = SGD(lr=0.03, decay=1e-6,  momentum=0.9,  nesterov=True)
from keras.optimizers import Adam
adam = Adam(lr=0.0003)


from keras.callbacks import History 
history = History()


model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#features = np.load('QCD_signal.npy')

features_train, labels_train, features_test, labels_test = load_datasets3()
#features, 

# the actual training
model.fit(features_train, labels_train, validation_split=0.1, nb_epoch=1000, batch_size=10000, callbacks=[history], sample_weight = None )
score = model.evaluate(features_test, labels_test, batch_size=10000)

