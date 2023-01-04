from tensorflow.keras import Sequential

import tensorflow.keras.applications.vgg16 as v16
import tensorflow.keras.applications.resnet_v2 as rn2
import tensorflow.keras.applications.densenet as dn
import tensorflow.keras.applications.inception_resnet_v2 as irn2

from tensorflow.keras.preprocessing import image as tfimg
import tensorflow.keras.applications.vgg19 as v19

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
import numpy as np
#import random
#import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#from sklearn.decomposition import PCA
#from sklearn.model_selection import GridSearchCV
import os
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

files_folder = 'd:/Dane/diabetic-retinopathy-detection/unpacked/selected_for_POC'
labels_file = 'd:/Dane/diabetic-retinopathy-detection/unpacked/trainLabels.csv'

def load_model_vgg19():
    model = v19.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224,3))
    return model

def load_model_densenet169():
    model = dn.DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224,3))
    return model

def load_model_resnet2():
    model = rn2.ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224,3))
    return model    

def load_model_inceptionrenet2():
    model = irn2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224,3))
    return model    

def load_labels(input_file):
    labelsMap = {}
    labels = pd.read_csv(input_file)
    for index, row in labels.iterrows():
        label = row['level']
        if label > 0:
            label = 1
        labelsMap[row['image']] = label
    return labelsMap    

def get_files_list(images_folder):
    res = []
    for path in os.listdir(files_folder):
        full_file_path = os.path.join(files_folder, path)
        if os.path.isfile(full_file_path):
            res.append((path, full_file_path))
    return res    

def merge_with_labels(files_list, labels):
    result = []
    sorted_labels = []
    for (fn, fp) in files_list:
        fn_no_ext = fn.replace('.jpeg','')
        label = labels[fn_no_ext]
        result.append((fn, fp, label))
        sorted_labels.append(label)
    return (result, sorted_labels)    

def load_images(files_with_labels, preprocess_input_function):
    records = []
    preprocessed = []
    imgs = []
    for tup in files_with_labels:
        file_name, file_path, label = tup
        rawImage = tfimg.load_img(file_path, target_size=(224, 224))
        imgArr = tfimg.img_to_array(rawImage)
        imgs.append(imgArr)
        expanded = np.expand_dims(imgArr, axis=0)
        pre = preprocess_input_function(expanded)
        preprocessed.append(pre)
        records.append((label, file_path, file_name, rawImage, expanded, pre))   
        
    imgs_np = np.array(imgs)
    preprocessed_imgs_np = preprocess_input_function(imgs_np)
        
    return (records, preprocessed, imgs_np, preprocessed_imgs_np)

def prepare_data(files_folder, labels_file, preprocess_input_function):
    labels = load_labels(labels_file)
    files_list = get_files_list(files_folder)
    (files_and_labels, sorted_labels) = merge_with_labels(files_list, labels)
    (images, preprocessed, imgs_np, preprocessed_imgs_np) = load_images(files_and_labels)
    sorted_labels_np = np.array(sorted_labels)
    X_train, X_test, y_train, y_test = train_test_split(imgs_np, sorted_labels_np, test_size=0.33, random_state=42)
    
    X_train_pre = preprocess_input_function(X_train)
    X_test_pre = preprocess_input_function(X_test)

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    return (X_train_pre, X_test_pre, y_train_cat, y_test_cat)    

def define_new_model_bak(base_model):
    flatten_layer = layers.Flatten()
    layer_activation = "relu"
    dense_layer_1 = layers.Dense(100, activation=layer_activation)
    dense_layer_2 = layers.Dense(100, activation=layer_activation)
    dense_layer_3 = layers.Dense(100, activation=layer_activation)
    dense_layer_4 = layers.Dense(50, activation=layer_activation)
    dense_layer_5 = layers.Dense(50, activation=layer_activation)
    prediction_layer = layers.Dense(2, activation='softmax')
    
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3,
        dense_layer_4,
        dense_layer_5,
        prediction_layer
    ])
    
    return model   

def define_new_model(base_model):
    flatten_layer = layers.Flatten()
    layer_activation = "relu"
    dense_layer_1 = layers.Dense(6400, activation=layer_activation)
    dense_layer_2 = layers.Dense(256, activation=layer_activation)
    prediction_layer = layers.Dense(2, activation='softmax')
    
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    return model    

def prepare_data(files_folder, labels_file, preprocess_input_function):
    labels = load_labels(labels_file)
    files_list = get_files_list(files_folder)
    (files_and_labels, sorted_labels) = merge_with_labels(files_list, labels)
    (images, preprocessed, imgs_np, preprocessed_imgs_np) = load_images(files_and_labels, preprocess_input_function)
    sorted_labels_np = np.array(sorted_labels)
    X_train, X_test, y_train, y_test = train_test_split(imgs_np, sorted_labels_np, test_size=0.33, random_state=42)
    
    X_train_pre = preprocess_input_function(X_train)
    X_test_pre = preprocess_input_function(X_test)

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    return (X_train_pre, X_test_pre, y_train_cat, y_test_cat)    

def perform_test(model_loading_function, preprocess_input_function, files_folder, labels_file, trainable):
    X_train_pre, X_test_pre, y_train_cat, y_test_cat = prepare_data(files_folder, labels_file, preprocess_input_function)
    
    base_model = model_loading_function()
    
    base_model.trainable = trainable

    model = define_new_model(base_model)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20,  restore_best_weights=True)

    model.fit(X_train_pre, y_train_cat, epochs=100, validation_split=0.2, batch_size=16, callbacks=[es])
    
    results = model.evaluate(X_test_pre, y_test_cat, batch_size=16)

    print(results)

perform_test(load_model_vgg19, v19.preprocess_input, files_folder, labels_file, False)

#perform_test(load_model_vgg19, v19.preprocess_input, files_folder, labels_file, False)

#28/28 [==============================] - 3s 95ms/step - loss: 5.6776e-04 - accuracy: 1.0000 - val_loss: 1.2966 - val_accuracy: 0.7000
#17/17 [==============================] - 1s 75ms/step - loss: 1.1824 - accuracy: 0.6407

#perform_test(load_model_vgg19, v19.preprocess_input, files_folder, labels_file, True)

#28/28 [==============================] - 7s 246ms/step - loss: 0.6214 - accuracy: 0.6911 - val_loss: 0.5870 - val_accuracy: 0.7273
#17/17 [==============================] - 1s 72ms/step - loss: 0.6809 - accuracy: 0.6667
#[0.6809422969818115, 0.6666666865348816]

#perform_test(load_model_densenet169, dn.preprocess_input, files_folder, labels_file, False)

#17/17 [==============================] - 1s 71ms/step - loss: 0.6420 - accuracy: 0.6667
#[0.6420382261276245, 0.6666666865348816]

#perform_test(load_model_densenet169, dn.preprocess_input, files_folder, labels_file, True)

#perform_test(load_model_resnet2, rn2.preprocess_input, files_folder, labels_file, False)

#28/28 [==============================] - 4s 152ms/step - loss: 0.6208 - accuracy: 0.6911 - val_loss: 0.5885 - val_accuracy: 0.7273
#17/17 [==============================] - 2s 123ms/step - loss: 0.6365 - accuracy: 0.6667
#[0.6365066766738892, 0.6666666865348816]

#perform_test(load_model_resnet2, rn2.preprocess_input, files_folder, labels_file, True)

#perform_test(load_model_inceptionrenet2, irn2.preprocess_input, files_folder, labels_file, False)

#28/28 [==============================] - 4s 136ms/step - loss: 0.6193 - accuracy: 0.6911 - val_loss: 0.5875 - val_accuracy: 0.7273
#17/17 [==============================] - 2s 108ms/step - loss: 0.6771 - accuracy: 0.6667
#[0.6770616769790649, 0.6666666865348816]

#perform_test(load_model_inceptionrenet2, irn2.preprocess_input, files_folder, labels_file, True)