# Multi-modal Incremental Bayesian Network (MMIBN) for Open world User Identification

This repository contains the code and the libraries for open world user recognition using Multi-modal Incremental Bayesian Network, which is integrated on a Pepper robot. Please cite the papers which are described below in Files section if you are using this code.

## Files

The *RecognitionMemory.py* file contains the code for the Multi-modal Incremental Bayesian Network (MMIBN) with and without Online Learning (OL) for open world user identification as described in the paper:

Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, Tony Belpaeme, "Multi-modal Incremental Bayesian Network with Online Learning for Open World User Identification", Frontiers in Robotics and AI, Human-Robot Interaction, (in review).

*recognition-service* folder contains the RecognitionService that is to be uploaded on the robot (it can also be used remotely). This service is used for obtaining multi-modal information from the user: face similarity scores, gender, age, height estimations of the user through NAOqi modules (ALFaceDetection and ALPeopleDetection, NAOqi 2.4 and 2.5 - works on both NAO and Pepper). This information is used by RecogniserMemory, amulti-modal incremental Bayesian network (MMIBN) with option for online learning (evidence based updating of likelihoods) for reliable recognition in open-world identification. Please cite the paper mentioned above. 

RecognitionMemory is integrated with RecognitionService which uses NAOqi to get recognition information. However, it can be integrated with other recognition software (see the comments in the code).

The *recognitionModule.py* contains the RecognitionModule which allows online user recognition using Pepper robot (SoftBank Robotics Europe), as used in the experiments in the paper (it can be used locally or remotely):

Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, Tony Belpaeme, 2018, "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction Social Robots in the Wild workshop.

*util* folder contains instructions to install libraries necessary for RecognitionMemory and the compiled libraries for NAOqi. Use README\_SHORT instructions to use the already compiled libraries provided in the folder (works for both Naoqi 2.4 and 2.5), otherwise, you can compile your libraries according to README\_FULL instructions.

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
**NOTE: If it is the first time running the code, uncomment line 264 ( self.cleanDB() ). Then comment it for the next times.**

Touch the left hand of Pepper robot to start the code.

## Usage without tablet interaction for HRI in recognitionModule: "Silent" mode

In line 118 in recognitionModule.py, set self.isTabletInteraction to False.

This mode does not need tablet interaction with the robot (i.e. the name is not requested from the user for confirmation, and the user does not enroll).

1. To start the recognition, call function:

```
    recogniseSilent()
```

2. If the person is *NOT previously enrolled*, add the person to the dataset: 

```
    addPersonManually(p_name, p_gender, p_age, p_height)
```

If the estimated recognition results would like to be used as the "true values" of the recognition, the function can be called as:

```
    addPersonManually(p_name, self.RB.recog_results[1][0], self.RB.recog_results[2][0], self.RB.recog_results[3][0])
```

**THIS FUNCTION SHOULD BE CALLED BEFORE confirmRecognitionSilent IFF THE PERSON IS NOT PREVIOUSLY ENROLLED**


3. To confirm the recognition, call: 

```
    confirmRecognitionSilent()
```

## Cross-validation

Use function runCrossValidation in RecognitionMemory with specified parameters for cross validation from recognition results on file. Recognition results are obtained from running the cross-validation on the (Pepper) robot for IMDB experiments.

## Cross-validation on the (Pepper) robot

Use function runCrossValidationOnRobot in RecognitionMemory with specified parameters for cross validation from recognition results on the robot (simulating HRI scenario with images from IMDB dataset).

## Revert to last saved state

In case of any erroneous recognition, the database can be reverted to the *LAST* (the one before the current one) recognition state. 
Uncomment the line 267: self.RB.revertToLastSaved(isRobot=True)
If a robot is being used for recognition, isRobot = True, otherwise, False.

## Using another robot or another identifier

Using another robot or using other identifier for the given modalities (face, gender, age, height) is definitely possible! You would need to modify the "FUNCTIONS FOR THE ROBOT" section in RecognitionMemory.py file (lines 3062- 3280).

## License

This dataset is released under GNU General Public License. A copy of this license is included with the code.

## Contact

For more details, see the paper "Multi-modal Open-Set Person Identification in HRI" by Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, Tony Belpaeme, available [http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf](here). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.
