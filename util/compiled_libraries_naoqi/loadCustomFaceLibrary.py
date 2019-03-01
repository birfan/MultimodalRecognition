#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""File for loading the custom face detection library to use image for recognition. Put this file
under /home/nao/dev/lib/ and add the library libfacedetection_2_5_2_44_pepper.so under the same folder
Link to get the library: http://protolab.aldebaran.com:9000/protolab/facedetection_custom/tree/master
Modify autoload.ini file in /home/nao/naoqi/preferences/ and 
under the [python] line add the path to this python file

Example:
...
[python]
#the/full/path/to/your/python_module.py   # load python_module.py
/home/nao/dev/lib/loadCustomFaceLibrary.py

...

This will exit the current face detection library and load the custom library at startup."""

import qi
import sys

if __name__ == "__main__":
    session = qi.Session()
    try:
        session.connect("tcp://127.0.0.1:9559")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    face = session.service("ALFaceDetection")
    launcher = session.service("ALLauncher")
    try:
        face.exit()
    except:
        pass
    face_lib_file = "/home/nao/dev/lib/libfacedetection_2_5_2_44_pepper.so"
    launcher.launchLocal(face_lib_file)
