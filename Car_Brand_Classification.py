# import the libraries as shown below
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

# Dataset paths
train_path = 'Datasets/train'
valid_path = 'Datasets/test'

# Import the ResNet50 library as shown below and add preprocessing layer to the front of ResNet50
# Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE +
                  [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False

# useful for getting number of output classes
folders = glob('Datasets/train/*')

# our layers - you can add more if you want
x = Flatten()(resnet.output)    # flatten the output from resnet

# Adding output layer with 3 output nodes as we have 3 classes.
# len(folders) = 3 and output layer activation function is softmax
# Doing all this to last layer which is x
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
# model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
# Here we are doing Data Augmentation
# Recaling is because all image values needs to be set between 0 and 1 so deviding bt 255
# By Scaling Computation will be easy for a machine because valuse will be 0 to 1 only.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Do not perform Data augmenttion on Test data
# Only Scaling needs to be performed.
test_datagen = ImageDataGenerator(rescale=1./255)

# Calling the augmentation function to perform augmentation on train data.
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')
print(training_set)
# Call Scaling function and give test dataset.
test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')


# Train/fit the model and assign model to variable r.
# Run the cell. It will take some time to execute
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save it as a h5 file
model.save('model_resnet50.h5')
y_pred = model.predict(test_set)

# take maximum of 3 values and assign 1 to that class
y_pred = np.argmax(y_pred, axis=1)

print(y_pred)


# load model
model = load_model('model_resnet50.h5')


# Inference
img = image.load_img('Datasets/Test/lamborghini/11.jpg',
                     target_size=(224, 224))

# convert in to a numpy array
x = image.img_to_array(img)
print(x.shape)
x = x/255  # normalize

x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
img_data.shape

model.predict(img_data)
a = np.argmax(model.predict(img_data), axis=1)

if a == 0:
    print("Audi")
elif a == 1:
    print("Lamorghini")
elif a == 2:
    print("Mercedes")
