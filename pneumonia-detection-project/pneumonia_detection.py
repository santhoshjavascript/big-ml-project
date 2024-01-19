#ProjectGurukul
#load all required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Input,Conv2D,Flatten,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

#import below library only if you are using google colab
#from google.colab.patches import cv2_imshow

#train and test directory path
train_dir= "xray/train/"
test_dir= "xray/test/"
im_size= 200
batch_size= 32

def dir_img_lab(dir):
  #fetch all the normal xray images and labels from given directory
  norm_img = glob.glob(dir+"NORMAL/*.jpeg")
  norm_labels = np.array(['normal']*len(norm_img))

  #fetch all the pneumonia xray images and labels from given directory
  pnm_img = glob.glob(dir+"PNEUMONIA/*.jpeg")
  pnm_labels = np.array(list(map(lambda x: x.split("_")[1],pnm_img)))

  return norm_img,norm_labels,pnm_img,pnm_labels
  
#get the normal and pneumonia images for training and testing 
#and also get the labels
trn_norm_img,trn_norm_lab,trn_pnm_img,trn_pnm_lab= dir_img_lab(train_dir) 
tst_norm_img,tst_norm_lab,tst_pnm_img,tst_pnm_lab= dir_img_lab(test_dir)

def get_x(files):
  #create a numpy array of the shape
  #(number of images, image size , image size, 1 for grayscale channel ayer)
  #this will be input for model
  train_x = np.zeros((len(files), im_size, im_size,1), dtype='float32')
  
  #iterate over img_file of given path
  for i, img_file in enumerate(files):
    #read the image file in a grayscale format and convert into numeric format
    #resize all images to one dimension i.e. 200x200
    img = cv2.resize(cv2.imread(img_file,cv2.IMREAD_GRAYSCALE),((im_size,im_size)))
    #reshape array to the train_x shape
    #1 for grayscale format
    img_array = np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0)
    train_x[i] = img_array.reshape(img_array.shape[1],img_array.shape[2],1)
  
  return train_x

#pass the normal and pneumonia images of training and testing sets
trn_norm_x= get_x(trn_norm_img)
trn_pnm_x= get_x(trn_pnm_img)
tst_norm_x= get_x(tst_norm_img)
tst_pnm_x= get_x(tst_pnm_img)

print("train normal array shape :",trn_norm_x.shape)
print("train pneumonia array shape :",trn_pnm_x.shape)
print("\ntest normal array shape :",tst_norm_x.shape)
print("test pneumonia array shape :",tst_pnm_x.shape)

#append pneumonia array to normal array and
#append pneumonia labels to normal labels of training and testing
x_train = np.append(trn_norm_x,trn_pnm_x,axis=0)
y_train = np.append(trn_norm_lab,trn_pnm_lab)

x_test = np.append(tst_norm_x,tst_pnm_x,axis=0)
y_test = np.append(tst_norm_lab,tst_pnm_lab)

#This will be the target for the model.
#convert labels into numerical format
encoder = OneHotEncoder(sparse=False)
y_train_enc= encoder.fit_transform(y_train.reshape(-1,1))
y_test_enc= encoder.fit_transform(y_test.reshape(-1,1))

#Image augmentation using ImageDataGenerator class
train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

#generate images for training sets 
train_generator = train_datagen.flow(x_train, 
                                     y_train_enc, 
                                     batch_size=batch_size)

#same process for Testing sets also by declaring the instance
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(x_test, 
                                   y_test_enc, 
                                   batch_size=batch_size)

#Input with the shape of (200,200,1)
inputt = Input(shape=(im_size,im_size,1))

#convolutional layer with maxpooling
x=Conv2D(16, (3, 3),activation='relu',strides=(1, 1),padding='same')(inputt)
x=Conv2D(32, (3, 3),activation='relu',strides=(1, 1),padding='same')(x)
x=MaxPool2D((2, 2))(x)

x=Conv2D(16, (3, 3),activation='relu',strides=(1, 1),padding='same')(x)
x=Conv2D(32, (3, 3),activation='relu',strides=(1, 1),padding='same')(x)
x=MaxPool2D((2, 2))(x)

x=Conv2D(16, (3, 3),activation='relu',strides=(1, 1),padding='same')(x)
x=Conv2D(32, (3, 3),activation='relu',strides=(1, 1),padding='same')(x)
x=MaxPool2D((2, 2))(x)

#Flatten layer
x=Flatten()(x)
x=Dense(50, activation='relu')(x)

#Dense (output) layer with the shape of 3(total number of labels)
outputt=Dense(3, activation='softmax')(x)

#create model class with inputs and outputs
model= Model(inputs=inputt,outputs=outputt)
#model summary
model.summary()

#epochs for model training and learning rate for optimizer
epochs = 50
learning_rate = 1e-3 #0.001

#using Adam optimizer to compile or build the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=["accuracy"])

#fit the training generator data and train the model
hist = model.fit(train_generator,
                 steps_per_epoch= x_train.shape[0] // batch_size,
                 epochs= epochs,
                 validation_data= test_generator,
                 validation_steps= x_test.shape[0] // batch_size)

#Accuracy Graph
plt.figure(figsize=(8,6))
plt.title('Accuracy scores')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.show()
 
#Loss Graph
plt.figure(figsize=(8,6))
plt.title('Loss value')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

#Save the model for prediction
model.save("s2s")

#ProjectGurukul Pneumonia Detection
model = load_model("s2s")
labels = ['bacteria','normal','virus']

#confusion matrix
y_pred = model.predict(x_test)
#transforming label back to original
y_pred = encoder.inverse_transform(y_pred)

#matrix of Actual vs Prediction data
c_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
plt.title('Confusion matrix',fontsize=14)
sns.heatmap(
    c_matrix, xticklabels=labels,yticklabels=labels,
    fmt='d', annot=True,annot_kws={"size": 14}, cmap='Reds')
plt.xlabel("Predicted",fontsize=12)
plt.ylabel("Actual",fontsize=12)
plt.show()

for i in range(230,280,3):
  pred = model.predict(x_test)[i]

  #Display the x-ray image
  cv2.imshow("ProjectGurukul",x_test[i])

  #uncomment below line only when working on google colab
  #cv2_imshow(x_test[i])
  
  print("Actual :",y_test[i]," Predicted :",labels[np.argmax(pred)])
