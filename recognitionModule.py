# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright 2017 Bahar Irfan                                                                            #
#                                                                                                        #
#  This script allows to run the RecognitionMemory using Multi-Modal Incremental Bayesian Network        #
#  (MMIBN) for person recognition on Pepper robot (Naoqi 2.4 and 2.5).                                   #
#                                                                                                        #
#  This is the script used for collecting the user study dataset in Plymouth University                  # 
#  presented in the paper, please cite the paper if using this code:                                     #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #
#  Usage: Touch Pepper's left hand to start the module.                                                  #
#                                                                                                        #
#  RecognitionModule, RecognitionMemory and each script in this project is under the GNU General Public  #
#  License.                                                                                              #
#========================================================================================================#

import qi
import stk.runner
import stk.events
import stk.services
import stk.logging

import time
import functools

import pandas
import ast

import RecognitionMemory

import os.path

# TODO remove debug decorator
@RecognitionMemory.for_all_methods(RecognitionMemory.print_function_name)
@qi.multiThreaded()
class RecognitionModule(object):
   
    APP_ID = "RecognitionModule"

    @qi.nobind
    def __init__(self, qiapp):
        # generic activity boilerplate
        self.qiapp = qiapp
        self.versionV = "1.0"
        self.events = stk.events.EventHelper(qiapp.session)
        self.s = stk.services.ServiceCache(qiapp.session)
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)
        self.parameters = ["name", "gender", "age", "height"]
        self.recog_folder = "Experiment/"
        # self.face_db = "faceDB"

        self.isUnknown = False
        self.isAddPersonToDB = False
        self.identity_est = ""

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def initSystem(self):
        image_path = "/home/nao/.local/share/PackageManager/apps/recognition-service/html/temp.jpg"
        
        self.RB = RecognitionMemory.RecogniserBN()

        self.isDBinCSV = True
        self.isMultipleRecognitions = False
        self.numMultRecognitions = 3

        self.RB.setFilePaths(self.recog_folder)
        self.RB.setSessionConstant(isDBinCSV = self.isDBinCSV, isMultipleRecognitions = self.isMultipleRecognitions, defNumMultRecog = self.numMultRecognitions)

        # NOTE: Uncomment this to clean the files and face recognition database
        # self.cleanDB()

        # NOTE: Uncomment this to revert the files to the previous recognition state.
        # self.RB.revertToLastSaved(isRobot=True)

        self.loadDB(self.RB.db_file)
        self.running = True
        self.timer = 60*30

    @qi.bind(returnType=qi.Map(qi.String, qi.Float), paramsType=(qi.String, qi.List(qi.String), qi.List(qi.Float), qi.Float, qi.Float, qi.String, qi.Float, qi.Float, qi.Float, qi.List(qi.String)))
    def get_identity(self, face_name, face_ids, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence, timestamp):
        """Returns the most probable face among all the given face.
        """
        self.set_face_recog_results(face_name, face_ids, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence, timestamp)
        _, _, name, quality = self.recogniseSilent()
        return {name: quality}

    @qi.bind(returnType=qi.String, paramsType=None)
    def learn(self, face_name, face_ids, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence, timestamp, is_new):
        # set data for face
        self.set_face_recog_results(face_name, face_ids, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence,
                                    timestamp)
        self.isUnknown = is_new  # decide if face must be added to DB or just refreshed
        if self.isUnknown:
            self.addPersonUsingRecogValues(face_name)
            self.RB.setPersonToAdd(self.person)
        else:   # just update DB with known person
            self.getPersonFromDB(face_name)

        self.confirmRecognitionSilent()
        return self.person[0]

    @qi.bind(returnType=qi.Bool, paramsType=None)
    def isInitializing(self):
        return self.RB.isInitializing()

    @qi.nobind
    def recogniseSilent(self):
        self.identity_est, quality_estimate = self.RB.recognise_mem() # get the estimated identity from the recognition network
        print "time for recognition:" + str(time.time()-self.recog_start_time)
        if self.identity_est:
            if self.identity_est == self.RB.unknown_var:
                if self.RB.isInitializing() and self.isPersonInDB(self.RB.face_recog_name ):
                    identity_name = self.RB.face_recog_name # take the name of the primary person in FR results.. Might be changed
                else:
                    identity_name = ""  # BN does not agree with FR results
            else:
                identity_name = self.RB.names[self.RB.i_labels.index(self.identity_est)]
        else:
            print "all images are discarded"
            identity_name = ""

        print "isRegistered : " + str(self.isRegistered) + ", id estimated: " + self.identity_est + " id name: " + identity_name
        return self.isRegistered, self.identity_est, identity_name, quality_estimate

    @qi.nobind
    def addPersonManually(self, p_name, p_gender, p_age, p_height):
        self.person[0] = str(self.num_db + 1)
        self.person[1] = p_name
        self.person[2] = p_gender
        self.person[3] = p_age
        self.person[4] = p_height
        self.person[5] = [self.getTime()]
        self.isAddPersonToDB = True
        self.isRegistered = False
        print "Adding person to the DB:", self.person
        self.RB.setPersonToAdd(self.person)

    @qi.nobind
    def addPersonUsingRecogValues(self, p_name):
        """Add a person using the estimated recognition results as the "true values" of the recognition.
        NOTE: recognise() should be called before calling this function!!"""
        self.addPersonManually(p_name, self.RB.recog_results[1][0], self.RB.recog_results[2][0], self.RB.recog_results[3][0])

    @qi.nobind
    def confirmRecognitionSilent(self):
        self.RB.confirmPersonIdentity(id=self.person[0], name=self.person[1], is_known=(not self.isUnknown)) # save the network, analysis data, csv for learning
        if self.isAddPersonToDB:
            self.loadDB(self.RB.db_file)

    @qi.nobind
    def getTime(self):
        numeric_day_name = int(time.strftime("%u")) # Numeric representation of the day of the week (1 (for Monday) through 7 (for Sunday))
        current_time = time.strftime("%T") #"%H:%M:%S" (21:34:17)
        day = time.strftime("%d")
        month = time.strftime("%B")
        year = time.strftime("%Y")
        date_today = [current_time, numeric_day_name, day, month, year]

        return date_today

    @qi.nobind
    def updateDB(self, csv_file, p_name):
        """This function updates the database csv file with the updated time of the person seen"""
        time_p = self.db_df.loc[self.db_df['name'] == p_name, 'times'].iloc[0]
        time_p.append(self.getTime())
        self.db_df.loc[self.db_df['name'] == p_name, 'times'].iloc[0] = time_p
        with open(csv_file, 'w') as fd:
            self.db_df.to_csv(fd, index=False, header=True)

    @qi.nobind
    def isPersonInDB(self, p_name):
        p_name = self.changeNameLetters(p_name)
        return p_name in self.db_df.name.values

    @qi.nobind
    def loadDB(self, csv_file):
        if os.path.isfile(csv_file):
            self.num_db = sum(1 for line in open(csv_file)) - 1
            self.db_df = pandas.read_csv(csv_file, dtype={"I": object}, converters={"times": ast.literal_eval})
        else:
            self.cleanDB()

    @qi.nobind
    def getPersonFromDB(self, p_name):
        print "in getPersonFromDB, p_name is :" + str(p_name)
        print "database is: " + str(self.db_df)

        self.person[0] = str(self.db_df.loc[self.db_df['name'] == p_name, 'id'].iloc[0])
        self.person[1] = p_name
        self.person[2] = self.db_df.loc[self.db_df['name'] == p_name, 'gender'].iloc[0]
        try:
            self.person[3] = self.db_df.loc[self.db_df['name'] == p_name, 'birthYear'].iloc[0]
        except:
            self.person[3] = self.db_df.loc[self.db_df['name'] == p_name, 'age'].iloc[0]
        self.person[4] = self.db_df.loc[self.db_df['name'] == p_name, 'height'].iloc[0]
        self.person[5] = self.db_df.loc[self.db_df['name'] == p_name, 'times'].iloc[0]
        print "The person is :",self.person
    
    @qi.bind(returnType=qi.Void, paramsType=None)
    def cleanDB(self):
        self.RB.resetFiles()
        self.num_db = 0

    @qi.bind(returnType=qi.Void, paramsType=[])    
    def stop(self):
        "stop"
        
        self.logger.info("RecognitionModule stopped by user request.")
        self.running = False
        self.RB.saveBN()
        self.qiapp.stop()
        
    @qi.nobind
    def on_stop(self):
        "Cleanup"
        self.stop()
        self.logger.info("RecognitionModule finished.")

    @qi.nobind
    def set_face_recog_results(self, face_name, face_ids, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence, timestamp):
        assert len(face_ids) == len(face_confidences)
        face_data = [recognition_accuracy, [[id, conf] for id, conf in zip(face_ids, face_confidences)]]
        gender_data = [gender, gender_confidence]
        age_data = [long(age), age_confidence]
        height_data = [height, height_confidence]
        self.RB.set_face_recog_results([face_data, gender_data, age_data, height_data, timestamp], face_name)


if __name__ == "__main__":
    stk.runner.run_service(RecognitionModule)



