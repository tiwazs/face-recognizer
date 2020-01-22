# Face_Recognizer

## Real time recognizer using Insightface.

### Introduction
###### With the boom of machine learning and the data science, face recognition algorithms have gain immense strenght for many types of applications, like monitoring, surveilance, and even accesibility and security.
###### This software is able to detect of every face in an image or a video, and, extracting certain biometric characteristics of the faces and comparing them to the biometric data on a database of registered users to recognize them.

#### Requirements and Instalation
Python >=3.5
Numpy
OpenCV >=4

###### Apache MXNet
https://mxnet.apache.org/get_started/?version=v1.5.1&platform=linux&language=python&environ=pip&processor=gpu

>>pip3 install mxnet-cuXXX

Note: the number at the end is the cuda version. if the cuda version installed is cuda 10.1 then you should use pip3 install mxnet-cu101

###### Insightface
https://github.com/deepinsight/insightface

>>pip3 install insightface
###### In case that you want to install from source to triain a custom model follow the instructions on the repository.
###### In case of wanting to use another deep learning library as base on the repository there's a link to versions of insightface on Tensorflow, pytorch, caffe etc.

### Usage
###### Nota: Esta implementación emplea GPU tanto para la detección como para el reconocimiento. es posible que la GPU no sea capaz de cargar ambos modelos, o que no se disponga de GPU en absoluto, para esto se debe modificar en el archivo processing_dataset.py en las líneas 111 (detección) y 115 (reconocimiento) ctx_id de 0 a -1 para que funcione con CPU solamente, y en el archivo face_recognition_v2.py en las líneas 181 y 185. 

#### Register
###### There are 2 codes for the recognition. register.py is used to register users on the database.

>>python3 register.py -n [Nombre]

###### After running this code a video sequence will start, the user should type the key "S" to take a picture to register. 4 or 5 pictures should be fine. after finishing type "Q" and the data will be processed and saved into de database.Right now the data is saved to a file, as a future improvement a database software could be used.

#### Recognizer
###### The file face_recogmition_v2.py opens a real time video sequence and for each frame will detect the faces in sight and will get their information to compare with the info in the database and apply the recognition.
###### Run with the command:
>>python3 face_recognition_v2.py

### Future improvements
###### Implementation of the algorithms in a specific application. A case could be in monitoring the people that enter a place, where we can automaticly register new folks to analize their behavior and identify them, be it on a store to track frequent buyers or as surveilance of suspicious people.
###### Apply the system so it doesn't require manual register and to do it automaticly when someone uknown enters the scene.
###### Implementation of an anti-spoofing system. Currently this algorithm is not able to differentiate between a picture, a video, or if it's a person on live cam. Implementing an anti-spoofing system the algorithms could be used in security and access.

