Instructions on how to make RecognitionMemory work on Nao/Pepper:

=====================================================

NAOqi FOR OpenNAO:


*   Download OpenNAO OS VirtualBox and C++ NAOqi cross toolchain for Linux 32 from Software (OR use the libnaoqi_files folder provided instead of the toolchain) in 


    http://community.aldebaran.com


*   Setup OpenNAO using VirtualBox


CONFIGURE GCC TO C++11 and C++14: 


*    In OpenNAO:

```
    $ mkdir src

    $ mkdir dev
```

On local computer:


*   Get gcc-5.3.0


*   Copy the folder to OpenNAO:

```
    $ scp -P 2222 -r gcc-5.3.0/ nao@localhost:/home/nao/src/
```

On OpenNAO:

```
    $ cd src/gcc-5.3.0

    $ mkdir build && cd build

    $ ../configure --enable-languages=c,c++

    $ sudo make -j8

    $ sudo make install exec_prefix=/usr/local
```

*   Export the local library path to update the gcc

```
    $ export "LD_LIBRARY_PATH=/usr/local/lib"
```

On robot:

*   Change the name of the libstdc++.so files

```
    $ mkdir dev && cd dev

    $ mkdir bin && mkdir lib

```

On OpenNAO:

*   Send the libraries from OpenNAO to NAO: 

```
    $ cd /usr/local/lib

    $ scp libstdc++.so* nao@robotIP:/home/nao/dev/src/lib
```

On robot:

```
    $ nano ~/.bash_profile
```

*   Add the following line at the bottom

```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nao/dev/src/lib
```

UPDATE NUMPY:

On OpenNAO:

*   Uninstall numpy:

```
    $ sudo pip uninstall numpy
```

*   Install numpy 1.8.1 (or the version in the robot if >1.8.1):

```
    $ sudo pip install 'numpy==1.8.1'
```

*   Install pandas (requires numpy >=1.8.0):

```
    $ sudo pip install pandas
```

On PC:

*   Copy pandas and pytz folders from OpenNAO to your PC:

```
    $ scp -P 2222 -r nao@localhost:/usr/lib/python2.7/site-packages/pandas .
    $ scp -P 2222 -r nao@localhost:/usr/lib/python2.7/site-packages/pytz .
    $ scp -P 2222 -r nao@localhost:/usr/lib/python2.7/site-packages/numpy .
```
*   Copy the folders from pc to the robot under the source folder of the code:

```
    $ scp -r pandas/ nao@ROBOT_IP:path_to_folder
    $ scp -r pytz/ nao@ROBOT_IP:path_to_folder
    $ scp -r numpy/ nao@ROBOT_IP:path_to_folder
```

Example path_to_folder: /home/nao/dev/src/pepper/

INSTALL PYAGRUM (requires cmake >=3.1.0, numpy >=2.8.1, gcc>= 5.3.0)

On OpenNao:

*   Download and install cmake 3.1.0. From https://cmake.org/files/v3.1/ install cmake-3.1.0.tar.gz.

```
    $ ./bootstrap && make && make install
```

*   In aGrUM/acttools/builder.py function getForMakeSystem(current, target)  change the following line to (add ../../   in front of wrappers) 

```
    line += " -C ../../wrappers/pyAgrum"
```

```
    $ cd aGrUM
    $ cmake .. -DCMAKE_CXX_COMPILER="/usr/local/bin/c++" -DCMAKE_C_COMPILER="/usr/local/bin/gcc" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0 -s -mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -s -mtune=atom -mssse3 -mfpmath=sse" -DUSE_SWIG=OFF -DUSE_NANODBC=OFF -DFOR_PYTHON3=OFF
```

*   Copy pyAgrum from OpenNAO to PC:

```
    $ scp -P 2222 -r nao@localhost:/usr/lib/python2.7/site-packages/aGrUM/wrappers/pyAgrum .
```

*   Copy pyAgrum from PC to the robot:

```
    $ scp -r pyAgrum/ nao@ROBOT_IP:path_to_folder
```


On robot:

```
    $ nano ~/.bash_profile
```

*   Add the following line at the bottom

```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_folder
```

UPDATE ALFACEDETECTION LIBRARY TO RECOGNISE/LEARN FACES FROM FILE:

On PC:

*   Download the ALFaceDetection library according to the version of your robot from: http://protolab.aldebaran.com:9000/protolab/facedetection_custom/tree/master

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

NOTE: THE COMPILED VERSION OF PANDAS, PYTZ AND PYAGRUM IS IN UTIL FOLDER. The folders are compiled for NAOqi "2.5.5.5" which has numpy version "1.8.1", pandas version "0.20.1", pytz version "2017.2", pyAgrum version "0.11.2.9".


