from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import json
from django.conf import settings



def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def verifyFace(img1, img2,vgg_face_descriptor):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(settings.BASE_DIR+'/static/dataset/testing/%s' % (img1)))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(settings.BASE_DIR+'/static/dataset/victims/%s' % (img2)))[0,:]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    #euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    epsilon1 = 0.30
    epsilon2 = 120
    cosineSame = epsilon1 - cosine_similarity

    print("similarity:", cosineSame)
    return cosineSame

#img1 is the tested image path, and dataset needs to be json.
def matchFaces(img1, dataset):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

#you can download the pretrained weights from the following link 
#https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
#or you can find the detailed documentation https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
    model.load_weights(settings.BASE_DIR+'/static/model/vgg_face_weights.h5')
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    top10 = {}
    top10[0] = 100000000
    for item in dataset:
        img2 = dataset[item]["path"]
        victimID = dataset[item]["id"]
        similarity = verifyFace(img1, img2,vgg_face_descriptor)
        if similarity>0:
            if len(top10)<11:
                top10[victimID] = similarity
            else:
                del top10[findMin(top10)]
                top10[victimID] = similarity
    del top10[0]
    for key in top10.keys():
        print(key)
    return top10.keys()

def findMin(l):
    mini = 0
    for i in l:
        if l[i]<l[mini]:
            mini = i
    return mini

def test(img):
    victims_file = open(settings.BASE_DIR+'/static/model/victims')
    json1_str = victims_file.read()
    dataset = json.loads(json1_str)
    faceList = matchFaces(img, dataset)
    print(faceList)
    return faceList



test("/Users/siyuzhu/Public/Rice University/HackRice/ADrop/mysite/static/dataset/testing/2.jpg")




