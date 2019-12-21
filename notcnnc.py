# Convolutional Neural Network



# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#extra layer
# classifier.add(Convolution2D(32,3,3,activation='relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(400, activation = 'relu'))#128,relu
classifier.add(Dropout(0.3))
classifier.add(Dense(400,activation = 'relu'))# 128,relu
classifier.add(Dropout(0.3))
classifier.add(Dense(4, activation = 'softmax'))#4,sigmoid

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'sparse')

# classifier.fit_generator(training_set,
#                          samples_per_epoch = 8000,
#                          nb_epoch = 5,
#                          validation_data = test_set,
#                          nb_val_samples = 2000)
classifier.load_weights('navin_1.h5')

import numpy as np
from keras.preprocessing import image
import matplotlib.image as mpimg

test_img = image.load_img('aa.jpg',target_size=(64,64))

test_img = np.expand_dims(test_img,axis=0)

img_class = classifier.predict_classes(test_img)

# c,d = classifier.evaluate_generator(training_set,1797)
e,f = classifier.evaluate_generator(test_set,607)
# # i = classifier.predict_generator(test_set,607)

# # print(confusion_matrix(test_set.classes,i))
# print("Loss:",e)
print("Accuracy of test:",f*100)
# print(img_class)
# print("Train:",d*100.0)
# print("Test",f*100.0)




prediction = img_class[0]
classname = img_class[0]
training_set.class_indices
if classname<=0:
    prediction ='cat'
elif classname>0 and classname<=1:
    prediction='dog'
elif classname >1 and classname<=2:
    prediction ='horses'
else:
    prediction='humans' 
print(prediction)
print(classname)


# print("class:",classname)

# img = test_img.reshape((28,28)) 
# img = test_img.transpose()
# print(img)

# print("Accuaracy",scores[1]*100)
# plt.imshow(img)


# Convert to PIL Image
import cv2


import cv2
face_cascade = cv2.CascadeClassifier('model-haar/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('aa.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display thnce output

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = ((int) (img.shape[1]/2 - 268/2), (int) (img.shape[0]/2 - 36/2))
fontScale              = 5
fontColor              = (255,255,255)
lineType               = 2
# cv2.putText(img,prediction, 
#     bottomLeftCornerOfText,  
#     font, 
#     fontScale,cnnc
#     fontColor,
#     lineType)
# cv2.imshow('img', img)
# cv2.waitKey()
b,g,r = cv2.split(img)       # get b,g,r
img = cv2.merge([r,g,b]) 
imgplot = plt.imshow(img)
plt.title(prediction)
plt.show()




from sklearn.metrics import confusion_matrix
predictions = classifier.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

print(confusion_matrix(test_set.classes,predicted_classes)
)



# Load the cascade




# training_set.class_indices
# if result[0][0]>=0.5:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
# print(prediction)
