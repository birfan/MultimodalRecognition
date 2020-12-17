# Install libraries on local computer (Ubuntu 16.04):

* python version: 2.7.12

```
    $ sudo apt-get install build-essential
    $ sudo apt-get install python-pip

    $ sudo apt-get install python-numpy==1.11.0 cython
    $ sudo apt-get install python-pandas==0.17.1
    $ sudo apt-get update
    $ sudo apt-get upgrade

    $ sudo -H pip install pyagrum==0.12.0.3
```

* If you will be running experiments on Pepper/NAO robots, you need to install qi library. Otherwise, you can remove import qi and the robot functions from RecognitionMemory.py. The python-SDKs for Python 2.7 for Naoqi 2.4 and 2.5 are provided in naoqi\_sdks folder.

```
    $ nano ~/.bashrc
```

* Add the following line to the end of the file:

For NAOqi 2.4:

```
    $ export PYTHONPATH=${PYTHONPATH}:/path-to-MultimodalRecognition/util/naoqi_sdks/pynaoqi-python2.7-2.4.3.28-linux64/
```

For NAOqi 2.5:
```
    $ export PYTHONPATH=${PYTHONPATH}:/path-to-MultimodalRecognition/util/naoqi_sdks/pynaoqi-python2.7-2.5.5.5-linux64/lib/python2.7/site-packages
```

* Source the bashrc file.
```
    $ source ~/.bashrc
``` 

* Check installation:

```
$ python

    >> import numpy
    >> import pandas
    >> import pyAgrum
    >> import qi
```
