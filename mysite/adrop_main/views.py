from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.http import HttpResponse
from pymongo import MongoClient

from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import json
from django.conf import settings
import matplotlib.pyplot as plt
from keras.preprocessing import image

import os


client = MongoClient()
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

@csrf_exempt
def compareAndSave(request):
    data = request.FILES['imageFile']
    img = Image.open(data)
    img.save(settings.BASE_DIR+'/static/dataset/testing/0.jpg')
    return JsonResponse({'error':False, 'message':"Uploaded successfully"})

def compare(request):
    if os._exists(settings.BASE_DIR+'/static/dataset/record.json'):
        os.remove(settings.BASE_DIR+'/static/dataset/record.json')
    open(settings.BASE_DIR + '/static/dataset/record.json', "w")
    result = test("0.jpg")
    #return JsonResponse(result)
    r = json.dumps(result)
    loaded_r = json.loads(r)

    with open(settings.BASE_DIR+'/static/dataset/record.json', "w") as f:
        json.dump(loaded_r, f)
    return JsonResponse(loaded_r)

def index(request):
    return render(request,settings.BASE_DIR+"/static/index/html/index.html")

def logonValidation(request):
    accountName = request.FILES['login_username']
    password = request.FILES['login_password']
    result = dict()
    result["value"] = "false"
    return JsonResponse(request,result)

def uploadImage(request):
    return render(request, settings.BASE_DIR + "/static/uploadimage/html/imageupload.html")

def showImage(request):
    return render(request, settings.BASE_DIR + "/static/showimage/html/showimage.html")

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

def verifyFace(img1, img2, vgg_face_descriptor):
    img1_representation = vgg_face_descriptor.predict(
        preprocess_image(settings.BASE_DIR + '/static/dataset/testing/%s' % (img1)))[0, :]
    img2_representation = vgg_face_descriptor.predict(
        preprocess_image(settings.BASE_DIR + '/static/dataset/victims/%s' % (img2)))[0, :]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    epsilon1 = 0.30
    epsilon2 = 120
    cosineSame = epsilon1 - cosine_similarity
    print("similarity:", cosineSame)
    return cosineSame

# img1 is the tested image path, and dataset needs to be json.
def matchFaces(img1, dataset):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # you can download the pretrained weights from the following link
    # https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    # or you can find the detailed documentation https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
    model.load_weights(settings.BASE_DIR + '/static/model/vgg_face_weights.h5')
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    top10 = {}
    top10[0] = 100000000
    for item in dataset:
        img2 = dataset[item]["path"]
        victimID = dataset[item]["id"]
        similarity = verifyFace(img1, img2, vgg_face_descriptor)
        if len(top10) < 11:
            top10[victimID] = similarity
        else:
            del top10[findMin(top10)]
            top10[victimID] = similarity
    del top10[0]
    newDict = dict()
    for key in top10.keys():
        jsonResult = findJson(key, dataset)
        if jsonResult:
            print("Person Found:", jsonResult)

        else:
            print("Sorry, no record")
        subDict = dict()
        subDict["id"]=key
        subDict["path"] = jsonResult["path"]
        newDict[key] = subDict
    return newDict

def findJson(key, dataset):
    for item in dataset:
        if dataset[item]["id"] == str(key):
            return dataset[item]

    return None



def findMin(l):
    mini = 0
    for i in l:
        if l[i] < l[mini]:
            mini = i
    return mini


def test(img):
    victims_file = open(settings.BASE_DIR + '/static/model/victims')
    json1_str = victims_file.read()
    dataset = json.loads(json1_str)
    faceList = matchFaces(img, dataset)
    print(faceList)
    return faceList
