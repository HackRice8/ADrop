# ADrop
This is a repository for Hackrice8

## Deep Face
This project uses DeepFace to do the one-shot face recognition, which can match two pictures and judge if they are the same person. 
In folder python, vgg_face.py, it can preprocess the input images and do verification.



## Inspiration
* Make Sense of Data track
uses some application of data science or machine learning
* Indeed Challenge
innovative projects and solutions that tackle societal problems

## What it does
Family members cannot get in contact with each other after natural disasters
Disaster victims might be hospitalized without identification
reconnect. 
first responders upload images and general information about disaster victims
family members search for their loved ones by uploading photos
facial-recognition matches the uploaded photos
return top 10 possible matches, user selects correct profile

## How we built it
### Deep Learning for Faces Match
#### Requirements
python=3.6.6

Matplotlib=2.2.2

tensorflow=1.7.0

keras=2.1.5

Cython=0.28.5
#### Pre-trained Model
Download pre-trained model [here](https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view)
* Step 1: 
Transform images to 
2622 dimensional 
vectors
* Step 2: 
Deep Learning
Training and 
Inference
* Step 3: 
Vector similarity:
- Cosine distance
* Step 4: 
Verify Matches:
Match = Distance less than threshold
* Step 5: 
Results:
Top 10 most similar 
profiles

### Build up the web application
* Step 1: 
Upload Image
* Step 2: 
Run the facial matching algorithm back-end
* Step 3: 
Return top-10 most similar people
* Step 4: 
Fetch information and give feed back


## What's next for Reconnect
* Areas for improvement
more accurate facial recognition
although our system deploys near state-of-the-art algorithmic architecture, we recognize that rapid advancements in technology may produce better infrastructure in the near future

* Future implementations
ability to be notified of new matches
store user-input images in a separate database ⇒ run matching algorithm for newly uploaded profiles ⇒ email/phone alert when new matches are found


