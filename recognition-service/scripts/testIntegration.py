#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

#========================================================================================================#
#  Copyright 2017 Bahar Irfan                                                                            #
#                                                                                                        #
#  Example for usage of RecognitionMemory with another interface (For Pepper robot interface,            #
#  use recognitionModule instead)                                                                        #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #                 
#  RecognitionService, RecognitionMemory and each script in this project is under the GNU General Public #
#  License.                                                                                              #
#========================================================================================================#

import RecognitionMemory
from datetime import datetime
import time

if __name__ == "__main__":
    start_time = time.time()
    RB = RecognitionMemory.RecogniserBN()
    robot_ip = "10.0.1.1"
    RB.connectToRobot(robot_ip, port=9559, useSpanish = False, isImageFromTablet = True, isMemoryOnRobot = False)
    isMemoryRobot = True # True if the robot with memory is used (get this from the days maybe?)
    isRegistered = True # False if register button is pressed (i.e. if the person starts the session for the first time)
    isAddPersonToDB = False # True ONLY IF THE EXPERIMENTS ARE ALREADY STARTED, THE BN IS ALREADY CREATED, ONE NEW PERSON IS BEING ADDED!FOR ADDING MULTIPLE PEOPLE AT THE SAME TIME, DELETE RecogniserBN.bif FILE INSTEAD!!!
    isDBinCSV = True # True if using DB as a CSV file, other option is not implemented.
    isMultipleRecognitions = True # True if multiple image are used for single recognition, False for single image detection.
    numMultRecognitions = 3 # Default number of images to be used for multiple image recognition.
    person = []

    # set session constants (doesn't change throughout the experiment - needs to be called once for the experiment)
    RB.setSessionConstant(isMemoryRobot = isMemoryRobot, isDBinCSV = isDBinCSV, isMultipleRecognitions = isMultipleRecognitions, defNumMultRecog = numMultRecognitions)

    # TODO: take a picture and send to robot!

    # set session variables (needs to be called before each person is recognised)
    RB.setSessionVar(isRegistered = isRegistered, isAddPersonToDB = isAddPersonToDB)

    identity_est = RB.recognise_mem() # get the estimated identity from the recognition network
    p_id = None
    isRecognitionCorrect = False
    if isMemoryRobot:
        if isRegistered:
            if identity_est != '0':
                # TODO: ask for confirmation of identity_est on the tablet (isRecognitionCorrect = True if confirmed) 
                isRecognitionCorrect = True # True if the name is confirmed by the patient
                
    if isRecognitionCorrect:
        RB.confirmPersonIdentity(p_id = identity_est) # save the network, analysis data, csv for learning and picture of the person in the tablet
    else:
        # TODO: take confirmation from the tablet
        # TODO: getPersonFromDB to set isAddPersonToDB (True, if person does not exist)
        if isAddPersonToDB:
            cur_date = datetime.now()
            person = ["5", "Dolores Abernathy", "Female", 25, 175, [cur_date]]
            p_id = '5'
            isRegistered = False
            RB.setPersonToAdd(person)
        else:
            p_id = '3'
        
        RB.confirmPersonIdentity(p_id = p_id)
    total_time = time.time() - start_time
    print total_time
    
