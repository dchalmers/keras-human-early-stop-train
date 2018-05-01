Interactive Early Stopping of Keras Training
===

Technique to enable user to cleanly early stop Keras training on epoch end by pressing 'Enter'

## Overview
* currently only two ways to 'early stop' Keras batch training
- use the early_stopping callback which stops on certain metrics (e.g. validation accuracy doesn't improve by a fixed delta)
- hit Ctrl-C
* the former isn't very interactive or dynamic, different hyperparameters may lead to wish to could stop after few or more epochs whilst model is running
* the latter doesn't allow the epoch to finish cleanly and unless you handle the interrupt won't allow you to run weight-saving or other clean-up code
* manual early stopping via keypress has been mentioned: https://github.com/keras-team/keras/issues/2161



