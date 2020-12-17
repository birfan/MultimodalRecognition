# Multi-modal Incremental Bayesian Network (MMIBN) for Open world User Identification

This repository contains the code and the libraries for open world user recognition using Multi-modal Incremental Bayesian Network, which is integrated on a Pepper robot, as described in the following papers:

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction Social Robots in the Wild workshop.

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (under review), "Multi-modal Incremental Bayesian Network with Online Learning for Open World User Identification", ACM Transactions on Human-Robot Interaction (THRI).

[https://agrum.gitlab.io/pages/pyagrum.html](pyAgrum) library is used for implementing the Bayesian network structure in MMIBN:

 * Christophe Gonzales, Lionel Torti and Pierre-Henri Wuillemin (2017), "aGrUM: a Graphical Universal Model framework", International Conference on Industrial Engineering, Other Applications of Applied Intelligent Systems, Springer, [https://doi.org/10.1007/978-3-319-60045-1_20](DOI:10.1007/978-3-319-60045-1_20)

Please cite all the papers mentioned above if you are using MMIBN.

## Files

The *RecognitionMemory.py* file contains the code for the Multi-modal Incremental Bayesian Network (MMIBN) with and without Online Learning (OL) for open world user identification. For online learning, set isOnlineLearning to True (line 158), and update the optimised weights and quality of the estimation accordingly (see code).

*recognition-service* folder contains the RecognitionService that is to be uploaded on the robot (it can also be used remotely). This service is used for obtaining multi-modal information from the user: face similarity scores, gender, age, height estimations of the user through NAOqi modules (ALFaceDetection and ALPeopleDetection, NAOqi 2.4 and 2.5 - works on both NAO and Pepper). This information is used by RecogniserMemory, a multi-modal incremental Bayesian network (MMIBN) with option for online learning (likelihoods are updated with each user recognition) for reliable recognition in open-world identification. For NAOqi 2.9, see a version in branch jb/naoqi-2.9 (this version does not support saving images on tablet).

RecognitionMemory is integrated with RecognitionService which uses NAOqi to get recognition information. However, it can be integrated with other recognition software (see the comments in the code).

The *recognitionModule.py* contains the RecognitionModule which allows real-time user recognition using Pepper robot (SoftBank Robotics Europe), as used in the experiments in Irfan et al. (2018).

*util* folder contains instructions to install libraries necessary for RecognitionMemory and the compiled libraries for NAOqi. Use README\_SHORT instructions to use the already compiled libraries provided in the folder (works for both Naoqi 2.4 and 2.5), otherwise, you can compile your libraries according to README\_FULL instructions.

**NOTE:** For the first few recognitions (e.g. 5 in the papers) defined by *num_recog_min* in *RecognitionMemory.py*, the user will be recognised as "unknown" to let BN build enough recognitions to provide better results and to decrease FAR. This parameter can be changed in line 122, to decrease or increase the number. If instead of "unknown", the face recognition estimate of the identity is desired to be used, set *isUseFaceRecogEstForMinRecog* to True (in line 123).

## Installation

 * Install compiled libraries to the robot using README\_SHORT file in util folder.

 * Install RecognitionService module to the robot according to README file in recognition-service module.

 * Send the recognition files to path\_to\_folder (same as the one in README\_SHORT file)

```
    $ scp recognitionModule.py nao@ROBOT_IP:path_to_folder
    $ scp RecognitionMemory.py nao@ROBOT_IP:path_to_folder
```

## Usage with tablet interaction as described in the HRI workshop paper

Start the code on the robot:

```
    $ mkdir Experiment
    $ cd path_to_folder
    $ python recognitionModule.py

```
**NOTE:** If it is the first time running the code, uncomment line 265 in *recognitionModule.py*:
```
    self.cleanDB()
```
Then comment it again before the next run.

Touch the left hand of Pepper robot to start the code.

## Usage without tablet interaction for HRI in recognitionModule: "Silent" mode

In line 118 in *recognitionModule.py*, set isTabletInteraction to False.

This mode does not need tablet interaction with the robot (i.e. the name is not requested from the user for confirmation, and the user does not enroll).

1. To start the recognition, call function:

```
    recogniseSilent()
```

2. If the person is *NOT previously enrolled*, add the person to the dataset: 

```
    addPersonManually(p_name, p_gender, p_age, p_height)
```

If the estimated recognition results would like to be used as the "true values" of the recognition (*recognise()* function should be called before calling this function!):

```
    addPersonUsingRecogValues(p_name)
```

**addPersonUsingRecogValues SHOULD BE CALLED BEFORE confirmRecognitionSilent IFF THE PERSON IS NOT PREVIOUSLY ENROLLED**


3. To confirm the recognition, call: 

```
    confirmRecognitionSilent()
```

## Cross-validation

Use function runCrossValidation in *RecognitionMemory.py* with specified parameters for cross validation from recognition results on file, which was used for the evaluations on the Multi-modal Long-Term User Recognition Dataset. More information about the dataset and the evaluations are available in the [https://github.com/birfan/MultimodalRecognitionDataset](MultimodalRecognitionDataset) repository.

## Revert to last saved state

In case of any erroneous recognition, the database can be reverted to the *LAST* (the one before the current one) recognition state. 
Uncomment the line 268 in *recognitionModule.py*: 

```
    self.RB.revertToLastSaved(isRobot=True)
```

If a robot is being used for recognition, set isRobot = True, otherwise, False.

## Using another robot or another identifier

Using another robot or using other identifier for the given modalities (face, gender, age, height) is definitely possible! You would need to modify the "FUNCTIONS FOR THE ROBOT" section in *RecognitionMemory.py* file (lines 3080-3298).

## License

This dataset is released under GNU General Public License v3.0. A copy of this license is included with the code.

## Contact

For more details, see Irfan et al. (2018; under review). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.

## Acknowledgments

We would like to thank Pierre-Henri Wuillemin for his substantial help with pyAgrum, and Jérôme Bruzaud and Victor Paléologue for adapting recognition-service and recognitionModule for NAOqi 2.9.

