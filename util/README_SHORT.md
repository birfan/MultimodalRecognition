# Instructions on installing libraries for RecognitionMemory on Nao/Pepper:

Example path_to_folder: /home/nao/dev/lib/

On robot:

*   Change the name of the libstdc++.so files in usr/lib

*   Add folders to send the new libraries

```
    $ mkdir dev && cd dev

    $ mkdir bin && mkdir lib && mkdir src

```

On PC:

*   Send the libstdc++.so from PC to NAO: 

```
    $ cd /usr/local/lib

    $ scp libstdc++.so* nao@robotIP:path_to_folder
```

*   Copy the folders from pc to the robot under the source folder of the code:

```
    $ scp -r pandas/ nao@ROBOT_IP:path_to_folder
    $ scp -r pytz/ nao@ROBOT_IP:path_to_folder
    $ scp -r numpy/ nao@ROBOT_IP:path_to_folder
    $ scp -r pyAgrum/ nao@ROBOT_IP:path_to_folder
```

On robot:

```
    $ nano ~/.bash_profile
```

*   Add the following line at the bottom

```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_folder


*   Save and close the file, and restart NAOqi

```
    $ nao restart
```

## UPDATE ALFACEDETECTION LIBRARY TO RECOGNISE/LEARN FACES FROM FILE:

On PC:

*   Download the ALFaceDetection library according to the version of your robot from (e.g. libfacedetection_2_5_2_44_pepper.so): http://protolab.aldebaran.com:9000/protolab/facedetection_custom/tree/master

*   Send the library and loadCustomFaceLibrary.py file to the robot to /home/nao/dev/lib folder (NOTE: CHANGE THE face_lib_file = "/home/nao/dev/lib/libfacedetection_2_5_2_44_pepper.so" LINE IN loadCustomFileLibrary.py file according to your robot)

```
    $ scp -r libfacedetection_2_5_2_44_pepper.so nao@ROBOT_IP:/home/nao/dev/lib/
    $ scp -r loadCustomFaceLibrary.py nao@ROBOT_IP:/home/nao/dev/lib/
```

On robot:

```
    $ nano naoqi/preferences/autoload.ini
```

*   Add the following line under [python]

```
/home/nao/dev/lib/loadCustomFaceLibrary.py
```

*   Save and close the file, and restart NAOqi

```
    $ nao restart
```

**NOTE: THE COMPILED VERSION OF PANDAS, PYTZ AND PYAGRUM IS IN UTIL FOLDER. The folders are compiled for NAOqi "2.5.5.5" which has numpy version "1.8.1", pandas version "0.20.1", pytz version "2017.2", pyAgrum version "0.11.2.9".**




