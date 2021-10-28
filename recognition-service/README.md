# Recognition Service for NAO and Pepper Robots: Installation Instructions

**Note that RecognitionService only works with NAOqi 2.4 and 2.5 libraries provided in the util folder.**

### On PC:

RecognitionService should be uploaded to the robot through Choregraphe:

* In Choregraphe: File->Open Project choose recognition-service/recognition-service.pml

* Connect to the robot. Go to the Robot Applications view, and click "Package and install current project to the robot" button (in the shape of Nao head with an arrow).

* Send the loadCustomFaceLibrary.py file to the robot:

```
    $ cd util/compiled_libraries_naoqi/

    $ scp loadCustomFaceLibrary.py nao@IP_ADDRESS_ROBOT:/home/nao/dev/lib/
```

* Add the library libfacedetection\_2\_5\_2\_44\_pepper.so (or the version for your robot) under the same folder.

### On the Robot:

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

### On PC:

Connect to the robot and clear the face recognition database:

```
    $ ssh nao@IP_ADDRESS_ROBOT

    $ qicli call ALFaceDetection.clearDatabase
```

## License

This project is released under GNU General Public License v3.0. A copy of this license is included with the code.

Cite the following if using this work:

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (2021), "Multi-modal Open World User Identification", Transactions on Human-Robot Interaction (THRI), 11 (1), ACM, [DOI:10.1145/3477963](https://doi.org/10.1145/3477963).

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction [Social Robots in the Wild workshop](http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf).

## Contact

For more details, see Irfan et al. (2018, 2021). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk (the most recent contact information is available at [personal website](https://www.baharirfan.com)).

