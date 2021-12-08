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
from xgboost import XGBClassifier

def build(properties):

    if(properties['model'] == 'dnn'):
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

    if(properties['model'] == 'bdt'):
        if not 'trees' in properties:
            raise ValueError('Property trees is required in the section training')
        if not 'max-depth' in properties:
            raise ValueError('Property max-depth is required in the section training')
        if not 'eta' in properties:
            raise ValueError('Property eta is required in the section training')
        if not 'workers' in properties:
            raise ValueError('Property workers is required in the section training')

        properties['trees'] = int(properties['trees'])
        properties['max-depth'] = int(properties['max-depth'])
        properties['eta'] = float(properties['eta'])
        properties['workers'] = int(properties['workers'])
        model = XGBClassifier(n_estimators=properties['trees'], use_label_encoder=False, max_depth=properties['max-depth'], eta=properties['eta'],
                              n_jobs=properties['workers'])

        return model
