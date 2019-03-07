# Recognition Service Installation Instructions

**Note that RecognitionService only works with NAOqi 2.4 and 2.5 libraries provided in the util folder.**

# On PC:

RecognitionService should be uploaded to the robot through Choregraphe:

* In Choregraphe: File->Open Project choose CRRobot/recognitionService/recognition-service.pml

* Connect to the robot. Go to the Robot Applications view, and click "Package and install current project to the robot" button (in the shape of Nao head with an arrow).

* Send the loadCustomFaceLibrary.py file to the robot:

```
    $ cd util/compiled_libraries_naoqi/

    $ scp loadCustomFaceLibrary.py nao@IP_ADDRESS_ROBOT:/home/nao/dev/lib/
```

* Add the library libfacedetection\_2\_5\_2\_44\_pepper.so (or the version for your robot) under the same folder.

# On the robot:

* Modify autoload.ini file in the robot /home/nao/naoqi/preferences/ and under the [python] line, add the path to this python file.

**Example:**
```
...
[python]
#the/full/path/to/your/python_module.py   # load python_module.py
/home/nao/dev/lib/loadCustomFaceLibrary.py
...
```

This will exit the current face detection library and load the custom library at startup.

# On PC:

Connect to the robot and clear the face recognition database:

```
    $ ssh nao@IP_ADDRESS_ROBOT

    $ qicli call ALFaceDetection.clearDatabase
```


