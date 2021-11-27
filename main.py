import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras import backend as k
# from tensorflow.keras.callbacks import Callback
# import matplotlib.pyplot as plt
# import gc
import time

# starts timer
start_time = time.time()

# prints tensorflow version
print("TensorFlow version:", tf.__version__)

# path to labels
data_dir = 'data/'
labels = []

# create labels for nvidia dataset
"""
with open('C:/Users/Games/Desktop/keras_nn_nvidia/data/steering.csv') as file:
    train_file = file.readlines()
    for index, line in enumerate(train_file):
        if index != 0:
            x = line[line.index(",") + 1: line.index(",", line.index(",") + 1, -1)]
            labels.append(float(x))
"""

# sorted picture names for nvidia dataset
"""
from os import listdir

mypath = 'data/images'

onlyfiles = [float(file[:file.index('.')]) for file in listdir(mypath)]

onlyfiles.sort()

new_list = []

for line in onlyfiles:
    line = str(line)
    new_list.append(f"{line[:-2]}.jpg")

print(new_list)
 """

# create labels for sully chen dataset
with open('data/data.txt') as file:
    train_file = file.readlines()
    hold1 = 0
    hold2 = 0
    counter = 0
    for line in train_file:
        # helps fix erroneous inputs of 0.00 from CSV file
        if float(line[(line.find(',') + 2): -1]) == 0:
            labels.append((((float(hold1 + hold2) / 2) + 540) / 1080))
            if counter % 2 == 0:
                hold1 = float((hold1 + hold2) / 2)
                counter += 1
            else:
                hold2 = float((hold1 + hold2) / 2)
                counter += 1
        elif counter % 2 == 0:
            hold1 = float(line[(line.find(',') + 2): -1])
            labels.append(((float(line[(line.find(',') + 2): -1]) + 540) / 1080))
            counter += 1
        else:
            hold2 = float(line[(line.find(',') + 2): -1])
            labels.append(((float(line[(line.find(',') + 2): -1]) + 540) / 1080))
            counter += 1

# creates datasets
train_dataset = image_dataset_from_directory(data_dir, labels=labels, image_size=(455, 256), seed=123,
                                             validation_split=0.2, subset="training")
val_dataset = image_dataset_from_directory(data_dir, labels=labels, image_size=(455, 256), seed=123,
                                           validation_split=0.2, subset="validation")

"""
# something to do with holding images in memory if possible - research what is happening in the 3 lines of code below
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(320).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
"""

# creates model
model = models.Sequential()
input_shape = (455, 256, 3)  # first layer needs input shape specified
model.add(tf.keras.layers.Rescaling(1./255, input_shape=input_shape))
model.add(layers.Conv2D(16, (5, 5), activation=None))
model.add(tf.keras.layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5), activation=None))
model.add(tf.keras.layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation=None))
model.add(tf.keras.layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10))
model.add(layers.Dense(1))

# defines optimizer, loss function, metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'],
    run_eagerly=True
)

# Instantiate an optimizer to train the model. NEW MATERIAL
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
# Instantiate a loss function.
loss_fn = tf.keras.losses.MeanSquaredError()

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.Accuracy()
val_acc_metric = tf.keras.metrics.Accuracy()

batch_size = 32

epochs = 6
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch + 1,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))


""" MODEL.FIT() FUNCTION IS BROKEN, DON'T USE. CAUSES OUT OF MEMORY ERROR.
# specifies optimizer and loss functions, as well as metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'],
    run_eagerly=True
)

# prints model summary
model.summary()


# custom callback helps deal with memory leak issue
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


# runs model from datasets
epochs = 6
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=ClearMemory()
)


# plots model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""
# saves trained model
model.save('model')


# stops timer, prints elapsed running time in seconds
print("--- {0:.2f} seconds ---".format((time.time() - start_time)))
