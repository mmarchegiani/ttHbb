import sys
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.activations import relu,linear,elu,sigmoid,tanh,softmax
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import *
from keras import metrics
from keras import losses
import keras.backend as K
from keras.models import model_from_yaml

def build(properties):

    if(properties['model'] == 'custom_model'):
        if not 'hidden-layers' in properties:
            raise ValueError('Property hidden-layers is required in the section training')
        if not 'neurons' in properties:
            raise ValueError('Property neurons is required in the section training')
        if not 'dropout-rate' in properties:
            raise ValueError('Property dropout-rate is required in the section training')
        if not 'output-dim' in properties:
            raise ValueError('Property output-dim is required in the section training')
        model = Sequential()

        properties['hidden-layers'] = int(properties['hidden-layers'])
        properties['neurons'] = int(properties['neurons'])
        #properties['output-dim'] = int(properties['output-dim'])
        properties['dropout-rate'] = float(properties['dropout-rate'])
        properties['learning_rate'] = float(properties['learning_rate'])
        if properties['optimizer'] is 'adam':
            properties['optimizer'] = keras.optimizers.Adam(learning_rate=properties['learning_rate'])
        elif properties['optimizer'] is 'sgd':
            properties['optimizer'] = keras.optimizers.SGD(learning_rate=properties['learning_rate'])

        if properties['hidden-layers'] > 0:
            model.add(Dense(units=properties['neurons'], input_dim=properties['input_dim'],
                            kernel_initializer=properties['kernel_init'], activation=properties['activation']))

        dropout=True

        if (properties['dropout-rate'] < 1e-8) : dropout=False

        for i in range(properties['hidden-layers'] - 1):
            if dropout : model.add(Dropout(properties['dropout-rate']))
            model.add(Dense(units=properties['neurons'],
                            kernel_initializer=properties['kernel_init'],
                            activation=properties['activation']))

        if dropout : model.add(Dropout(properties['dropout-rate']))
        model.add(Dense(units=properties['output-dim'], 
                        kernel_initializer=properties['kernel_init'],
                        activation=properties['last_activation']))
        #optimizer could be even customized
        model.compile(loss=properties['loss'],
                        optimizer=properties['optimizer'],
                        metrics=[properties['metrics']])
        print('model.summary()' + str(model.summary()))
        yaml_string = model.to_yaml()
        print('yaml_string'+ yaml_string)
        return model

    if(properties['model'] == 'dihiggs_model'):
        if not 'hidden-layers' in properties:
            raise ValueError('Property hidden-layers is required in the section training')
        if not 'neurons' in properties:
            raise ValueError('Property neurons is required in the section training')
        if not 'dropout-rate' in properties:
            raise ValueError('Property dropout-rate is required in the section training')
        if not 'output-dim' in properties:
            raise ValueError('Property output-dim is required in the section training')
        model = Sequential()

        opt = Adam(lr=0.0001)

        properties['hidden-layers'] = int(properties['hidden-layers'])
        properties['neurons'] = int(properties['neurons'])
        #properties['output-dim'] = int(properties['output-dim'])
        properties['dropout-rate'] = float(properties['dropout-rate'])

        if properties['hidden-layers'] > 0:
            model.add(Dense(units=properties['neurons'], input_dim=properties['input_dim'],
                            kernel_initializer=properties['kernel_init'], activation=properties['activation']))

        dropout=True

        if (properties['dropout-rate'] < 1e-8) : dropout=False

        for i in range(properties['hidden-layers'] - 1):
            if dropout : model.add(Dropout(properties['dropout-rate']))
            model.add(Dense(units=properties['neurons'],
                            kernel_initializer=properties['kernel_init'],
                            activation=properties['activation']))

        if dropout : model.add(Dropout(properties['dropout-rate']))
        model.add(Dense(units=256,
                        kernel_initializer=properties['kernel_init'],
                        activation=properties['activation']))

        if dropout : model.add(Dropout(properties['dropout-rate']))
        model.add(Dense(units=properties['output-dim'], 
                        kernel_initializer=properties['kernel_init'],
                        activation=properties['last_activation']))
        #optimizer could be even customized
        model.compile(loss=properties['loss'],
                      optimizer=opt,
                      metrics=[properties['metrics']])
        yaml_string = model.to_yaml()
        return model
