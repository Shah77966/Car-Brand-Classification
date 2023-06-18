# Image Classification using ResNet50

This README file provides an overview of the code for image classification using the ResNet50 model in TensorFlow/Keras. It explains the purpose of each section of the code and provides instructions on how to use it.

## Prerequisites
Before running the code, please ensure you have the following:

- Python installed (version 3.x).
- TensorFlow and Keras libraries installed.
- The dataset available in the specified directory structure.

## Dataset
The code assumes that you have a dataset organized in the following structure:

```
Datasets/
├── train/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    ├── class3/
    └── ...
```

Each class folder contains images corresponding to that class.

## Getting Started
1. Install the required libraries:

```shell
pip install tensorflow keras matplotlib
```

2. Import the necessary libraries:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
```

3. Specify the image size for resizing:

```python
IMAGE_SIZE = [224, 224]
```

4. Set the paths for the training and validation datasets:

```python
train_path = 'Datasets/train'
valid_path = 'Datasets/test'
```

5. Load the ResNet50 model with pre-trained weights and freeze the layers:

```python
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the existing weights
for layer in resnet.layers:
    layer.trainable = False
```

6. Determine the number of output classes:

```python
folders = glob('Datasets/train/*')
```

7. Build the model architecture by adding custom layers on top of the pre-trained ResNet50:

```python
x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=prediction)
```

8. Compile the model:

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

9. Set up data augmentation and image scaling for the training and testing datasets:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

10. Load the training and testing datasets using the data generators:

```python
training_set = train_datagen.flow_from_directory(
    'Datasets/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'Datasets/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

11. Train the model:

```python
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
```



12. Plot the training and validation loss:

```python
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
```

13. Plot the training and validation accuracy:

```python
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
```

14. Save the trained model:

```python
model.save('model_resnet50.h5')
```

15. Perform inference on a test image:

```python
# Load the saved model
model = load_model('model_resnet50.h5')

# Load and preprocess the test image
img = image.load_img('Datasets/Test/lamborghini/11.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# Make predictions
prediction = model.predict(img_data)
a = np.argmax(prediction, axis=1)

# Print the predicted class
if a == 0:
    print("Audi")
elif a == 1:
    print("Lamborghini")
elif a == 2:
    print("Mercedes")
```

Please make sure to replace the paths and class names according to your dataset. You can modify the code as needed to suit your specific requirements.

That's it! You can now use this README file as a guide to understand and run the image classification code using ResNet50.