# Instructions to Install Libraries for RecognitionMemory on Local Computer

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

* If you will be running experiments on Pepper/NAO robots, you need to install qi library. Otherwise, you can remove import qi and the robot functions from RecognitionMemory.py. The python-SDKs for Python 2.7 for NAOqi 2.4 and 2.5 are provided in naoqi\_sdks folder.

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

This work was tested on Ubuntu 16.04.

## License

This project is released under GNU General Public License v3.0. A copy of this license is included with the code.

Cite the following if using this work:

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (2021), "Multi-modal Open World User Identification", Transactions on Human-Robot Interaction (THRI), ACM, [DOI:10.1145/3477963](https://doi.org/10.1145/3477963).

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction [Social Robots in the Wild workshop](http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf).

 * Christophe Gonzales, Lionel Torti and Pierre-Henri Wuillemin (2017), "aGrUM: a Graphical Universal Model framework", International Conference on Industrial Engineering, Other Applications of Applied Intelligent Systems, Springer, [DOI:10.1007/978-3-319-60045-1_20](https://doi.org/10.1007/978-3-319-60045-1_20).

## Contact

For more details, see Irfan et al. (2018, 2021). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk (the most recent contact information is available at [personal website](https://www.baharirfan.com)).
