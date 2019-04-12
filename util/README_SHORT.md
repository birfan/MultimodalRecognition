# Instructions on installing libraries for RecognitionMemory on Nao/Pepper:

Example path\_to\_folder: /home/nao/dev/lib/

On robot:

*   Add folders to send the new libraries

```
    $ mkdir dev && cd dev

    $ mkdir lib

```

On PC:

*   Send the libstdc++.so from PC util folder to NAO: 

```
    $ scp libstdc++.so* nao@robotIP:path_to_folder
```

*   Copy the folders from pc to the robot under the source folder of the code:

```
    $ scp -r pandas/ nao@ROBOT_IP:path_to_folder
    $ scp -r pytz/ nao@ROBOT_IP:path_to_folder
    $ scp -r numpy/ nao@ROBOT_IP:path_to_folder
    $ scp -r pyAgrum/ nao@ROBOT_IP:path_to_folder
    $ scp -r stk/ nao@ROBOT_IP:path_to_folder
```

On robot:

```
    $ nano ~/.bash_profile
```

*   Add the following line at the bottom (Create a new file if it doesn't exist)

```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_folder:path_to_folder/pyAgrum
```

* Restart the robot (Turn it off and on again).

## UPDATE ALFACEDETECTION LIBRARY TO RECOGNISE/LEARN FACES FROM FILE:

On PC:

*   The compiled libraries for Naoqi (libfacedetection\_2\_4\_2\_25\_nao.so, libfacedetection\_2\_4\_2\_26\_pepper.so, libfacedetection\_2\_5\_2\_44\_pepper.so) are under util folder. If your robot requires another Naoqi version, you can download the ALFaceDetection library according to the version of your robot from: http://protolab.aldebaran.com:9000/protolab/facedetection_custom/tree/master

*   Send the library and loadCustomFaceLibrary.py file to the robot to /home/nao/dev/lib folder (NOTE: CHANGE THE face_lib_file = "path\_to\_folder/libfacedetection\_2\_5\_2\_44\_pepper.so" LINE IN loadCustomFileLibrary.py file according to your robot)

```
    $ scp -r libfacedetection_2_5_2_44_pepper.so nao@ROBOT_IP:path_to_folder
    $ scp -r loadCustomFaceLibrary.py nao@ROBOT_IP:path_to_folder
```

On robot:

```
    $ nano naoqi/preferences/autoload.ini
```

*   Add the following line under [python]

```
/home/nao/dev/lib/loadCustomFaceLibrary.py
```

* Restart the robot.

**NOTE: THE COMPILED VERSION OF PANDAS, PYTZ AND PYAGRUM IS IN UTIL FOLDER. The folders are compiled for NAOqi "2.5.5.5" which has numpy version "1.8.1", pandas version "0.20.1", pytz version "2017.2", pyAgrum version "0.11.2.9".**



