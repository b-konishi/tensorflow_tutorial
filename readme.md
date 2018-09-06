# Tensorflow Tutorial

## Environment
- venv
- pip

## pip-Package
- tensorflow
- matplotlib

(Following packages, to install scikit-learn)
- scipy
- pandas
- numpy

## caution
TensorBoard cannot function on venv.
When it uses, venv must be deactivated.

## Tensorboard histograms
Histogram can show the value of tensor in the valuable.
As histogram goes to front, time passes.
The horizontal axis indicates value of the tensor.
The value which focucing in your mouse indicates the value's frequency.
'''
tf.summary.histogram('name', name)
'''

