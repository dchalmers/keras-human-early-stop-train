'''
Demonstrate a technique to enable early stopping of training by user pressing 
'Enter'
Allows the current epoch to end cleanly before stopping training (vs catching Ctrl-C)
Allows human judgement per run to determine when accuracy has plateaued
(as opposed to specifying an early stopping loss accuracy metric)
Outline of technique:
1) create a global list var a_list to receive keyboard input
2) define a global function to add keyboard input to var
3) create a callback class which sub-classes 'on_epoch_end()' to end training if
global 'a_list' variable has a value (indicating user pressing 'Enter')
4) instantiate the callback to a variable and add it to the callback list of
the model
5) create a thread with the listener function
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import _thread

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
cutdown_factor = 10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

a_list = []

def input_thread(a_list):
    '''
    Worker thread to listen for input and set global semaphore if received
    input is line oriented, so the actual keypress being listened for is 'Enter'
    '''
    input()
    print('Input registered. Wait for end of epoch to finalize...')
    a_list.append(True)

# The data, split between train and test sets:
# cutdown to do fast testing of harness code
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cutpoint = x_train.shape[0] // cutdown_factor
x_train = x_train[:cutpoint, :, :, :]
y_train = y_train[:cutpoint]
cutpoint = x_test.shape[0] // cutdown_factor
x_test = x_test[:cutpoint, :, : :]
y_test = y_test[:cutpoint]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#start listening for input on a_list
_thread.start_new_thread(input_thread, (a_list,))


class enterkey_stop_train(Callback):
    '''
    create a callback class 
    sub-classes: on_epoch_end()
    print a message and set stop_training to True if keyboard input has been received
    '''
    def on_epoch_end(self, epoch, logs={}):
        if a_list:
            print('Epoch end with keypress. Stopping training...')
            self.model.stop_training = True

# instantiate the callback
stopcb = enterkey_stop_train()

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              callbacks=[stopcb],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        callbacks=[stopcb],  # callback to end training on enter
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
