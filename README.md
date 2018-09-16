# ADrop
This is a repository for Hackrice8

## Deep Face
This project uses DeepFace to do the one-shot face recognition, which can match two pictures and judge if they are the same person. 
In folder python, vgg_face.py, it can preprocess the input images and do verification.



## Inspiration (Hackathon)
* Make Sense of Data track
use some application of data science or machine learning
* Indeed Challenge
innovative projects and solutions that tackle societal problems

## What it does
Family members cannot get in contact with each other after natural disasters, and disaster victims might be hospitalized without identification. **reconnect.** bridges the gap: first responders upload images and general information about disaster victims, family members search for their loved ones by uploading photos of them, our facial-recognition system returns top 10 most similar profiles, and the user selects the correct match.


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
Transform images to 2622 dimensional vectors
* Step 2: 
Deep Learning Training and Inference
* Step 3: 
Calculate vector similarity (cosine distance)
* Step 4: 
Verify Matches (Match = Distance less than threshold)
* Step 5: 
Return results (top 10 most similar profiles)
* Step 6:
Build and connect to web application


## What's next for Reconnect
Area for improvement
* More accurate facial recognition- we recognize that rapid advancements in technology may produce better infrastructure in the near future

Future implementations
* Ability to be notified of new matches
** General setup: store user-input images in a separate database ⇒ run matching algorithm for newly uploaded profiles ⇒ email/phone alert when new matches are found

* Age-adjusting algorithm- find matches using an older image of the person (when they were younger)

