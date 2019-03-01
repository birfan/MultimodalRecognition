# Multi-modal Incremental Bayesian Network (MMIBN) for Open world User Identification

This repository contains the code and the libraries for open world user recognition using Multi-modal Incremental Bayesian Network, which is integrated on a Pepper robot. Please cite the papers which are described below in Files section if you are using this code.

## Files

The *RecognitionMemory.py* file contains the code for the Multi-modal Incremental Bayesian Network (MMIBN) with and without Online Learning (OL) for open world user identification as described in the paper:

Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, Tony Belpaeme, "Multi-modal Incremental Bayesian Network with Online Learning for Open World User Identification", Frontiers in Robotics and AI, Human-Robot Interaction, (in review).

*recognition-service* folder contains the RecognitionService that is to be uploaded on the robot (it can also be used remotely). This service is used for obtaining multi-modal information from the user: face similarity scores, gender, age, height estimations of the user through NAOqi modules (ALFaceDetection and ALPeopleDetection, NAOqi 2.4 and 2.5 - works on both NAO and Pepper). This information is used by RecogniserMemory, amulti-modal incremental Bayesian network (MMIBN) with option for online learning (evidence based updating of likelihoods) for reliable recognition in open-world identification. Please cite the paper mentioned above. 

RecognitionMemory is integrated with RecognitionService which uses NAOqi to get recognition information. However, it can be integrated with other recognition software (see the comments in the code).

The *recognitionModule.py* contains the RecognitionModule which allows online user recognition using Pepper robot (SoftBank Robotics Europe), as used in the experiments in the paper (it can be used locally or remotely):

Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, Tony Belpaeme, 2018, "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction Social Robots in the Wild workshop.

*util* folder contains instructions to install libraries necessary for RecognitionMemory and the compiled libraries for NAOqi.

## License

This dataset is released under GNU General Public License. A copy of this license is included with the code.

## Contact

For more details, see the paper "Multi-modal Open-Set Person Identification in HRI" by Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, Tony Belpaeme, available [http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf](here). For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.
