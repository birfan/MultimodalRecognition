# Multi-modal Incremental Bayesian Network (MMIBN) for Open world User Identification

This repository contains the code and the libraries for open world user recognition using Multi-modal Incremental Bayesian Network, which is integrated on a Pepper robot, as described in the following papers:

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (2021), "Multi-modal Open World User Identification", Transactions on Human-Robot Interaction (THRI), ACM, [DOI:10.1145/3477963](https://doi.org/10.1145/3477963).

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction [Social Robots in the Wild workshop](http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf).

[pyAgrum](https://agrum.gitlab.io/pages/pyagrum.html) library is used for implementing the Bayesian network structure in MMIBN:

 * Christophe Gonzales, Lionel Torti and Pierre-Henri Wuillemin (2017), "aGrUM: a Graphical Universal Model framework", International Conference on Industrial Engineering, Other Applications of Applied Intelligent Systems, Springer, [DOI:10.1007/978-3-319-60045-1_20](https://doi.org/10.1007/978-3-319-60045-1_20).

Please cite all the papers mentioned above if you are using MMIBN.

## Files

The *RecognitionMemory.py* file contains the code for the Multi-modal Incremental Bayesian Network (MMIBN) with and without Online Learning (OL) for open world user identification. 

For using online learning, use `setOnlineLearning()` function of *RecognitionMemory.py*. Update the optimised weights and quality of the estimation using `setOptimParams(isPatterned = False, isOnlineLearning=False)` function of *RecognitionMemory.py*. If the interaction will be on patterned times (e.g., 10 return visits to a rehabilitation session) use isPatterned=True, else it is False (e.g., for companion robot). If using online learning, use isOnlineLearning=True, else False.

*recognition-service* folder contains the RecognitionService that is to be uploaded on the robot (it can also be used remotely). This service is used for obtaining multi-modal information from the user: face similarity scores, gender, age, height estimations of the user through NAOqi modules (ALFaceDetection and ALPeopleDetection, NAOqi 2.4 and 2.5 - works on both NAO and Pepper). This information is used by RecogniserMemory, a multi-modal incremental Bayesian network (MMIBN) with option for online learning (likelihoods are updated with each user recognition) for reliable recognition in open-world identification. For NAOqi 2.9, see a version in branch `jb/naoqi-2.9` (this version does not support saving images on tablet).

RecognitionMemory is integrated with RecognitionService which uses NAOqi to get recognition information. However, it can be integrated with other recognition software (see the comments in the code).

The *recognitionModule.py* contains the RecognitionModule which allows real-time user recognition using Pepper robot (SoftBank Robotics Europe), as used in the experiments in Irfan et al. (2018).

*util* folder contains instructions to install libraries necessary for RecognitionMemory and the compiled libraries for NAOqi. Use README\_SHORT instructions to use the already compiled libraries provided in the folder (works for both Naoqi 2.4 and 2.5), otherwise, you can compile your libraries according to README\_FULL instructions.

**NOTE:** For the first few recognitions less than *num_recog_min* (e.g. 5 in the papers) in *RecognitionMemory.py*, the user will be recognised as "unknown" to let BN build enough recognitions to provide better results and to decrease FAR. This parameter can be changed using `setNumRecogMin(minrecog)` function of *RecognitionMemory.py*, to decrease or increase the number. If instead of "unknown", the face recognition estimate of the identity is desired to be used, use `setFaceRecogEstimateForMinRecog()` function of *RecognitionMemory.py*.

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

In line 118 in *recognitionModule.py*, set *isTabletInteraction* to *False*.

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

**`addPersonUsingRecogValues` SHOULD BE CALLED BEFORE `confirmRecognitionSilent` IFF THE PERSON IS NOT PREVIOUSLY ENROLLED**


3. To confirm the recognition, call: 

```
    confirmRecognitionSilent()
```

## Recognition Files

 * *db.csv*: This file contains the ID of the user (corresponding to the order of enrolment), the user's name, gender, height, the time of interaction when the user enrolled (*times*), and *occurrence* which corresponds to \[*number_occurrences_of_user*, *number_of_images_taken_while_enrolling*, *number_total_images_of_user*\]. The *occurrence* is \[0,0,0\] as default.

 * *RecogniserBN.bif*: Saved MMIBN file that contains states, prior and likelihoods of the nodes (*I*, *F*, *G*, *A*, *H*, *T*) for the model.

 * *InitialRecognition.csv*: Contains the estimated user identity (*I_est*) and the estimated properties (face (*F*), gender (*G*) and age (*A*)) when a user is seen (*N* is the counter of encounters, i.e., number of the recognition). Estimated user identity is based on the recognition module. *F* is in the format \[*confidence_score_of_face_recognition*, \[ \['user_id_2', *similarity_score*\] , \['user_id_5', *similarity_score*\], ...\]\] sorted in descending order of similarity score for all users in the face dataset. *G* is in the format \[*estimated_gender*, *confidence_score_of_gender_recognition*\]. Note that gender recognition is binary (*Male* or *Female*) due to the gender recognition algorithm in NAOqi. *A* is in the format \[*estimated_age*, *confidence_score_of_age_recognition*\]. The *H* is the estimated time in the format \[*estimated_height*, *confidence_score_of_height_estimation*\]. For NAOqi, the confidence score for *H* (0.08) is estimated from the standard deviation of height estimations (6.3 cm) found in Irfan et al. (2018). *T* is the time of interaction in the format, \['HH:MM:SS', 'Day_of_the_week'\], where Monday is day 1.

 * *RecogniserBN.csv*: If the user is new (i.e., the user is not previously encountered), the user is added to the face recognition database of NAOqi, and the face recognition, gender and age estimations are taken again on the same image. This file contains the corresponding *F*, *G*, and *A* parameters and the true identity of the user (*I*) and whether the user enrolled to the system in that encounter (*R* is 1) or not (*R* is 0). *H* and *T* are the same as in *InitialRecognition.csv*. If the user is not new, then the *N* entry for *InitialRecognition.csv* and *RecogniserBN.csv* will be the same.

 * *Analysis/Comparison.csv*: Compares user recognition model performance to face recognition. *I_real* is the true identity of the user, *I_est* is the estimated identity by the model, and *F_est* is the estimated identity by face recognition. *I_prob* is the posterior scores/probability scores for each identity starting with unknown user (ID 0) and the remaining IDs in ascending order (1,2,3..), *F_prob* is the face recognition similarity scores with the first entry 0 if highest similarity score is above that of the face recognition threshold (0.4 for NAOqi), and 0 otherwise. *Calc_time* is the time required for recognition (estimation and confirmation of identity) by the model. *R* represents whether the user is registered (enrolled), 0 if the user is already registered, 1 if the user is registering in the current recognition entry. *Quality* is the quality of the estimation (the difference between the highest posterior score and the second highest score, divided by the number of enrolled users). *Highest_I_prob* is the highest posterior (or probability) score for the model, and *Highest_F_prob* is the highest similarity score of face recognition.

 * *default*: The resulting face recognition database extracted from NAOqi.

 * *images*: Sorted images according to the recognition results. If the user is known and recognised correctly, the image will be under *Known_True*, or if that user is recognised as someone else, the image will be in *Known_False*, or if that user is recognised as a new user, the image will be in *Known_Unknown*; if the user is new, and recognised correctly as a new user, the image will be in *Unknown_True*, but if that user is recognised as a known user, the image will be in *Unknown_False*; if no face can be detected in the image, the image will be in *discarded*. Each image is named in the format *N*_*I*_*O*, where *N* is the number of recognition, *I* is the identity of the user for that fold, and *O* is the occurrence of that user. For instance, for the 40th recognition, the user 3 is seen for the second time, and incorrectly recognised as user 1, then the image *40_3_2.jpg* will be in *Known_False*.

## Cross-validation

Use function runCrossValidation in *RecognitionMemory.py* with specified parameters for cross validation from recognition results on file, which was used for the evaluations on the Multi-modal Long-Term User Recognition Dataset. More information about the dataset and the evaluations are available in the [MultimodalRecognitionDataset](https://github.com/birfan/MultimodalRecognitionDataset) repository.

## Revert to last saved state

In case of any erroneous recognition, the database can be reverted to the *last* (the one before the current one) recognition state. 
Uncomment the line 268 in *recognitionModule.py* (or run it separately): 

```
    self.RB.revertToLastSaved(isRobot=True)
```

To remove the feature to save the last files (to save on memory or during optimisation), use setSaveLastFiles(False) function of *RecognitionMemory.py*.

If a robot is being used for recognition, set isRobot = True, otherwise, False.

## Using another robot or another identifier

Using another robot or using other identifiers for the given modalities (face, gender, age, height) is definitely possible! You would need to modify the "FUNCTIONS FOR THE ROBOT" section in *RecognitionMemory.py* file (lines 3080-3298). However, the obtained biometric estimate needs to be in the same form (see *InitialRecognition.csv* file description above for formats).

## License

This project is released under GNU General Public License v3.0. A copy of this license is included with the code.

Cite the following if using this work:

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (2021), "Multi-modal Open World User Identification", Transactions on Human-Robot Interaction (THRI), ACM, [DOI:10.1145/3477963](https://doi.org/10.1145/3477963).

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction [Social Robots in the Wild workshop](http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf).

 * Christophe Gonzales, Lionel Torti and Pierre-Henri Wuillemin (2017), "aGrUM: a Graphical Universal Model framework", International Conference on Industrial Engineering, Other Applications of Applied Intelligent Systems, Springer, [DOI:10.1007/978-3-319-60045-1_20](https://doi.org/10.1007/978-3-319-60045-1_20).

## Contact

For more details, see Irfan et al. (2018, 2021). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk (the most recent contact information is available at [personal website](https://www.baharirfan.com)).

## Acknowledgments

We would like to thank Valerio Biscione for his valuable suggestions in the design of the MMIBN, Pierre-Henri Wuillemin for his substantial help with pyAgrum, and Jérôme Bruzaud and Victor Paléologue for adapting recognition-service and recognitionModule for NAOqi 2.9.

