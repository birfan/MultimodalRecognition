# Install libraries on local computer (Ubuntu 16.04):

* python version: 2.7.12

```
$ sudo apt-get install build-essential
$ sudo apt-get install python-pip

$ sudo apt-get install python-numpy==1.14.0 cython
$ sudo apt-get install python-pandas==0.17.1
$ sudo apt-get update
$ sudo apt-get upgrade

$ sudo -H pip install pyagrum==0.12.0.3
```

* Check installation:

```
$ python

>> import numpy
>> import pandas
>> import pyAgrum
```
