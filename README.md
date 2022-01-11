# ttHbb
Framework for accelerated ttH(bb) columnar analysis with Coffea (https://coffeateam.github.io/coffea/) on flat centrally produced nanoAOD samples.
## How to run
### Execution on local machine with Futures Executor
To run the preliminary version of the analysis script:
~~~
python runner.py --workflow dilepton --executor futures --samples datasets/baseline_samples.json --year 2017 --output test.coffea --DNN
~~~
### Output files
The output will be stored in two files: in `histograms/test.coffea` the histograms are saved in the `.coffea` format, while in `inputs/test.h5` the arrays are saved to be used as input for the DNN.

## DNN training
### Setup
Set the environment variable `NEW_REPO` to the directory containing `ttHbb/DNN`:

```
export NEW_REPO=/path/to/ttHbb/DNN
```
### Usage
- PREPROCESS of hdf file in order to create HDF python dataset to be used in the Deep Neural Network framework. In the following we split the dataset in train, test and validation:
~~~
python splitDataset.py -i /path/to/input/test.h5 -s 0.7:0.2:0.1
~~~
  --> this will save the train, test and validation datasets in `/path/to/input`
          
- TRAINING:
~~~
python Training.py -c /path/to/JobOption/NNconfig.cfg
~~~
  --> this will create all combination of different NN layout and produce separate naming folders where models, loss, training logs are saved

- EVALUATION:
~~~
python Evaluate.py -c /path/to/JobOption/NNconfig.cfg -p neu
~~~
  --> this will evaluate all the DNN models, created in the previous step
  
- PLOT:
Use the Jupyter notebook `ttHbb/DNN/Scripts/plot/plot_ttHbb.ipynb`

### DNN configuration

Settings of the DNN are cotrolled via the configuration file (`.cfg`). The same configuration file is used for the training (`Training.py`) and evaluation (`Evaluate.py`). Each configuration file has to contain ALL the parameters from this list. An example can be found in [NNRegressionExample.cfg](https://github.com/grossiM/DNN_neutrino_reco/blob/master/DNN_main/JobOption/NNRegressionExample.cfg).

| Parameter name  | Allowed values | Description
| :-----:         | :---:          | :----------------------------------------
| `output-folder` | string         | Absolute path to the output DNN model.
| `save-steps`    | 0, 1           | 0 means only the model from the last epoch will be saved, 1 means that models from each epoch will be saved.
| `data-train`    | string         | Absolute path to the training HDF dataset.
| `data-val`      | string         | Absolute path to the validation HDF dataset.
| `number-of-events` | integer     | Number of events from the training and the evaluation datasets to be processed. If set to -1, all the events are going to be processed.
| `training-variables` | string    | The list of the training features. Different features are separated by `,`, each item in the list has to correspond to a column name from the input HDF datasets.
| `training-labels`    | string    | The list of the training labels. Different labels are separated by `,`, each item in the list has to correspond to a column name from the input HDF datasets.
| `output-dim`         | integer   | Corresponds to the number of items in the `training-labels`.
| `selection`          | string    | Specifies the selection to be performed on the input HDF dataset prior to the training. The expression has to take the form supported by the [pandas.DataFrame.query](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html). In case one does not want to perform any selection, keyword `none` should be used.
| `scale-label`        | 0, 1      | Controls whether the label's scaling is applied before the training or not. If 1 is selected the label unscaling will automatically get applied after the evaluation.
| `optimizer` | `adam`, `SGD`, etc.  | Minimization/maximization algorithm.
| `loss` | `binary_crossentropy`, `mean_square_error`, etc.  | Loss function definition.
| `epochs`             | integer  | Number of epochs.
| `batch-size`         | integer  | Batch size.
| `metrics` | `binary_accuracy`, `categorical_accuracy`, etc.  | Characteristic variables to be monitored during the training.
| `model`         | `custom_model`, `dihiggs_model` | The ID of the DNN model type to be used. Currently two models are implemented in the framework: `custom_model` and `dihiggs_model`. `dihiggs_model` is meant for a more specific usecases and therefore the user is advized to use `custom_model`, which provides a skeleton for a DNN topology with equal number of nodes/neurons in each hidden layer. The type of the model determines the set of parameters needed in order to uniquely define the DNN topology. `custom_model` takes `hidden-layers`, `neurons`, `dropout-rate`, `activation`, `last-activation` and `kernel_init`. If the flexibility of the `custom_model` is insufficient, one can define their own model type with not too much effort in [Model.py](https://github.com/grossiM/DNN_neutrino_reco/blob/master/DNN_main/TrainingHandler/Model.py).
| `hidden-layers`      | integer   | Number of hidden layers.
| `neurons`            | integer   | Number of nodes/neurons in each hidden layer.
| `dropout-rate`       | float     | Dropout rate, no dropout rate is selected by setting it to 0.
| `activation`  | `relu`, `tanh`, etc.    | Activation function of the hidden layers.
| `last_activation`  | `sigmoid`, `linear`, etc.    | Activation function of the output layers.
| `kernel_init`      | `normal`, `identity`, etc.   | One of the keras methods for initialization of layer weights. Look [here](https://keras.io/api/layers/initializers/).
| `grid-search`      | string    | Any of the variables concerning the learning process can be optimized in a grid search optimization procedure. The parameters that are intended to be scanned in this procedure should be listed here, using `,` as the separator. Values of parameters that are going to be tested in the grid search are parsed through the appropriate argument (`neurons`, `hidden-layers`, etc.) using a `,` separated list instead of a single value. In case one wants to perform a single model training a dummy parameter still has to be passed into `grid-search`, but a single value for it can be specified as its value.
| `unfold`           | 0, 1      | If more than one parameter has been parsed to the `grid-search`, this parameter determines how the combinations are going to be formed. If it is 0, the number of values passed to each of the parameters being studies in the grid search has to be the same, because the combinations are formed from the parameter values which occupy the same sequential position in the `,` separated list for each parameter. If it is 1, the mumbers of parameter values can different, since all the possible combinations of parameters being provided are going to be examined.
| `data-eval`     | string        | Absolute path to the evaluation HDF dataset. The output HDF will inherit the name of the input, with addition of `.calibrated` to its base name.
| `output`       | string          | Absolute path to the folder to which the output HDF is going to be deposited.
| `model-of-interest` | `current_model_epochXXX` or `current_model`     | Specification of the model (epoch) which is going to be evaluated for each DNN layout from the training step. If `save-steps` has been set to 0, `model-of-interest` can only take `current_model`.
| `type`   | `binary`, `regression` | If the value is `binary`, ROC curves and AUC are evaluated for each DNN model. Output score is set to 0 or 1, based on the optimal working point extracted from the ROC curve.

