from tkinter import *
import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2
import os
import tkinter as tk
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

filter_size1 = 3 
num_filters1 = 32


filter_size2 = 3
num_filters2 = 32


filter_size3 = 3
num_filters3 = 64


fc_size = 128           


num_channels = 3


img_size = 32


img_size_flat = img_size * img_size * num_channels

img_shape = (img_size, img_size)


classes = ['Pneumonia','Normal','Covid']
num_classes = len(classes)
batch_size = 1
validation_size = .16
early_stopping = None

train_path = 'data/'
test_path = 'test/'



data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)
print("Size of:")
print("- Training-set: {}".format(len(data.train.labels)))
print("- Test-set: {}".format(len(test_images)))
print("- Validation-set: {}".format(len(data.valid.labels)))


images, cls_true  = data.train.images, data.train.cls

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels,    # Num. channels in prev. layer.
                   filter_size,                     # Width and height of each filter.
                   num_filters,                  # Number of filters.
                   use_pooling=True):     # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')


    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          
                 num_inputs,     
                 num_outputs,    
                 use_relu=True): 

    
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


layer_conv1   
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)
layer_conv2
layer_conv3

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)



cost = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session =  tf.compat.v1.Session()

session.run(tf.initialize_all_variables())
train_batch_size = batch_size



def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
   
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0


def optimize(num_iterations):
    
    global total_iterations

    
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

       
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

       
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        

        
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

  
    total_iterations += num_iterations

   
    end_time = time.time()

    
    time_dif = end_time - start_time

   
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


print(total_iterations)


##Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    
    
def plot_confusion_matrix(cls_pred):
    cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text and test metrics.
    print(cm)

    tp=cm[0][0]
    tn=cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
    fp=cm[0][1]+cm[0][2]
    fn=cm[1][0]+cm[2][0]

    tp_n=cm[1][1]
    tn_n=cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2]
    fp_n=cm[1][0]+cm[1][2]
    fn_n=cm[0][1]+cm[2][1]
    
    tp_c=cm[2][2]
    tn_c=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    fp_c=cm[2][0]+cm[2][1]
    fn_c=cm[0][2]+cm[1][2]
    
    
    print('\n***Test Metrics*** \n')
    precision = round(tp/(tp+fp)*100,2)
    recall = round(tp/(tp+fn)*100,2)
    specificity = round(tn/(tn+fp)*100,2)
    accuracy=round(((tp+tn)/15)*100,2)
    f1=round(2*precision*recall/(precision+recall),2)

    precision_n = round(tp_n/(tp_n+fp_n)*100,2)
    recall_n = round(tp_n/(tp_n+fn_n)*100,2)
    specificity_n = round(tn_n/(tn_n+fp_n)*100,2)
    accuracy_n=round(((tp_n+tn_n)/15)*100,2)
    f1_n=round(2*precision_n*recall_n/(precision_n+recall_n),2)

    precision_c= round(tp_c/(tp_c+fp_c)*100,2)
    recall_c = round(tp_c/(tp_c+fn_c)*100,2)
    specificity_c = round(tn_c/(tn_c+fp_c)*100,2)
    accuracy_c=round(((tp_c+tn_c)/15)*100,2)
    f1_c=round(2*precision_c*recall_c/(precision_c+recall_c),2)
    
    

    print("\t\t\tPrecision\tRecall\t\tSpecificity\tF1-score\tAccuracy\n")
    
    print('\tPneumonia\t{0}%\t\t{1}%\t\t{2}%\t\t{3}\t\t{4}%'.format(precision,recall,specificity,f1,accuracy))
    print('\tNormal\t\t{0}%\t\t{1}%\t\t{2}%\t\t{3}\t\t{4}%'.format(precision_n,recall_n,specificity_n,f1_n,accuracy_n))
    print('\tCovid\t\t{0}%\t\t{1}%\t\t{2}%\t\t{3}\t\t{4}%'.format(precision_c,recall_c,specificity_c,f1_c,accuracy_c))
    
    
    print("\n***Output for test images***")

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def print_validation_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    
    num_test = len(data.valid.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        
        j = min(i + batch_size, num_test)

    
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        

    
        labels = data.valid.labels[i:j, :]

       
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    print()
    msg = "True_values/Total_values:( {0} / {1} )"
    print(msg.format(correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    #  if show_example_errors:
    # print("Example errors:")
    #plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("\n***Confusion Matrix on Validation***")
        plot_confusion_matrix(cls_pred=cls_pred)


optimize(num_iterations=400)  # We performed 100 iterations above.
#print_validation_accuracy(show_example_errors=True)
print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)



def plot_conv_weights(weights, input_channel=0):
    
    w = session.run(weights)

   
    w_min = np.min(w)
    w_max = np.max(w)

   
    num_filters = w.shape[3]


    num_grids = math.ceil(math.sqrt(num_filters))
    
 
    fig, axes = plt.subplots(num_grids, num_grids)

   
    for i, ax in enumerate(axes.flat):
       
        if i<num_filters:
            
            img = w[:, :, input_channel, i]

           
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
       
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    plt.show()

def plot_conv_layer(layer, image):
   
    
    image = image.reshape(img_size_flat)

   
    feed_dict = {x: [image]}

   
    values = session.run(layer, feed_dict=feed_dict)

   
    num_filters = values.shape[3]

    
    num_grids = math.ceil(math.sqrt(num_filters))
  
    fig, axes = plt.subplots(num_grids, num_grids)

   
    for i, ax in enumerate(axes.flat):
       
        if i<num_filters:
           
            img = values[0, :, :, i]

           
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        
        ax.set_xticks([])
        ax.set_yticks([])
    
   
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(img_size, img_size, num_channels),
               interpolation='nearest')
    plt.show()
    '''plot_conv_weights(weights=weights_conv1)
    
    plot_conv_layer(layer=layer_conv1, image=image)
    plot_conv_weights(weights=weights_conv2, input_channel=0)
    plot_conv_layer(layer=layer_conv2, image=image)
    #session.close()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()'''

classes = { 0:'Pneumonia',
            1:'Normal',      
            2:'Covid'}

top=tk.Tk()
top.geometry('800x600')
top.title('PNEUMONIA AND COVID DETECTION')
top.configure(background='light blue')

label=Label(top,background='light blue', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image =cv2.imread(file_path)
    cv2.imshow("frame",image)
   

        
    image= cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR) / 255
    feed_dict_test = {
        x: image.reshape(1, img_size_flat),
        y_true: np.array([[2,1,0]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    sign=classes[test_pred[0]]
    print(sign)
    label.configure(foreground='#011638', text=sign)
    image1 = test_images[0]
    plot_image(image1)
     
    
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify X-ray Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an X-ray image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="PNEUMONIA AND COVID DETECTION",pady=20, font=('arial',20,'bold'))
heading.configure(background='light blue',foreground='black')
heading.pack()
top.mainloop()


