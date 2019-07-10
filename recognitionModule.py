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
    def __init__(self, qiapp):
        # generic activity boilerplate
        self.qiapp = qiapp
        self.versionV = "1.0"
        self.events = stk.events.EventHelper(qiapp.session)
        self.s = stk.services.ServiceCache(qiapp.session)
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)
        self.parameters = ["name", "gender", "age", "height"]
        self.person = ["","","",0,0,[]]
        self.counter = 0
        self.answer_counter = 0
        self.recog_folder = "Experiment/"
        self.face_db = "faceDB"

        self.r_ip = "127.0.0.1" #NOTE: This is not to be changed if the code is running locally on the robot. If using remotely, set this to correct value.
        
        self.isRegistered = True
        self.isUnknown = False
        self.isAddPersonToDB = False
        self.recog_end_time = time.time()
        self.recognised_people = []
        self.recognised_times = []
        
        self.timeoutPeriod = 90000 #Timeout period for waiting for input to the tablet in milliseconds
        
        self.imageShown = False
        self.webviewShown = False
        self.idleShown = False

        # Subscribe to left hand
        self.touch_hand_event = "HandLeftBackTouched"
        self.touch_hand = None
        self.id_hand = -1
        self.touch_hand = self.s.ALMemory.subscriber(self.touch_hand_event)
        self.id_hand = self.touch_hand.signal.connect(functools.partial(self.onTouchedHand, self.touch_hand_event))
        
        # Left bumper event initialisation (to disregard the recognition)
        self.touch_bumper_event = "LeftBumperPressed"
        self.touch_bumper = None
        self.id_bumper = -1
        self.bumperPressed = False

        # Activate picture shot
        self.is_camera_shooting = False
        
        # Head touched event initialisation
        self.touch_head_event = "FrontTactilTouched"
        self.touch_head = None
        self.id_head = -1
        self.headTouched = False
        
        # People detected event initialisation
        self.peopleDetectedEvent = "PeoplePerception/VisiblePeopleList"
        self.peopleDetected = None
        self.idPeopleDetected = -1
        
        self.person_id = None        
        self.heightPerson = None
        self.peopleDetectTimer = 10.0
                
        # Face detected event initialisation
        self.faceDetectedEvent = "FaceDetection/FaceDetected"
        self.faceDetected = None
        self.idFaceDetected = -1
        
        self.faceDetectTimer = 5.0
        
        self.hasAnalysedPerson = False
        self.isRecognitionOn = False
        self.identity_est = ""
        self.isAskedForReposition = False
        self.isNameAsked = False
        self.isRegistrationAsked = False
        self.isInputAsked = False
        self.isTabletInteraction = False # True if the interaction is from tablet, False otherwise

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def initSystem(self):
        image_path = "/home/nao/.local/share/PackageManager/apps/recognition-service/html/temp.jpg"
        
        self.RB = RecognitionMemory.RecogniserBN()

        self.isDBinCSV = True
        self.isMultipleRecognitions = False
        self.numMultRecognitions = 3
        self.hasAnalysedPerson = False

        self.RB.setFilePaths(self.recog_folder)
        self.RB.setSessionConstant(isDBinCSV = self.isDBinCSV, isMultipleRecognitions = self.isMultipleRecognitions, defNumMultRecog = self.numMultRecognitions)

        if os.path.isfile(self.recog_folder + self.face_db):
            self.RB.useFaceDetectionDB("faceDB")
        else:
            self.cleanDB()
            self.RB.resetFaceDetectionDB()
            self.RB.setFaceDetectionDB("faceDB")
           
        # NOTE: Uncomment this to clean the files and face recognition database
        self.cleanDB()

        # NOTE: Uncomment this to revert the files to the previous recognition state.
        # self.RB.revertToLastSaved(isRobot=True)

        self.loadDB(self.RB.db_file)
        self.running = True
        self.timer = 60*30

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onFaceDetected(self, strVarName, value): # TODO change function name ?
        """Modified version of RecognitionService onFaceDetected,i.e. picture is taken after a face is found, and face results are to be found from that picture"""
        self.recogniseSilent()

        if self.hasAnalysedPerson:
            self.hasAnalysedPerson = False
            self.RB.setPersonToAdd(self.person)

            self.confirmRecognitionSilent()

    @qi.bind(returnType=qi.AnyArguments, paramsType=[]) 
    def recogniseSilent(self):
        self.isUnknown = False
        self.isRegistered = True
        self.hasAnalysedPerson = False
        self.RB.setSessionVar(isRegistered=self.isRegistered, isAddPersonToDB=False)
        
        self.identity_est = self.RB.recognise_mem() # get the estimated identity from the recognition network
        print "time for recognition:" + str(time.time()-self.recog_start_time)
        if self.identity_est:
            self.hasAnalysedPerson = True
            if not self.s.ALBasicAwareness.isEnabled():
                self.s.ALBasicAwareness.setEnabled(True)
            if self.identity_est == self.RB.unknown_var:
                identity_name = self.RB.face_recog_name # take the name of the primary person in FR results.. Might be changed
                if not self.isNewFromName(identity_name): # TODO change to isPersonInDB(name)
                    self.getPersonFromDB(identity_name)
                else:
                    self.isUnknown = True
                    self.addPersonUsingRecogValues(identity_name)
                print "isRegistered : " + str(self.isRegistered) + ", id estimated: " + self.identity_est + " id name: " + identity_name

            else:
                identity_name = self.RB.names[self.RB.i_labels.index(self.identity_est)]
                print identity_name
                self.getPersonFromDB(identity_name)  # TODO FIX THIS !!
                print "isRegistered : " + str(self.isRegistered) + ", id estimated: " + self.identity_est + " id name: " + identity_name
                # self.s.ALMemory.raiseEvent("RecognitionResultsWritten", [self.isRegistered, self.identity_est, identity_name])

        else:
            print "all images are discarded"
            identity_name = ""

        self.subscribeToHead()
        self.detectPeople()

        return self.isRegistered, self.identity_est, identity_name
    
    @qi.bind(returnType=qi.Void, paramsType=[]) 
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
        
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def addPersonUsingRecogValues(self, p_name):
        """Add a person using the estimated recognition results as the "true values" of the recognition.
        NOTE: recognise() should be called before calling this function!!"""
        self.addPersonManually(p_name, self.RB.recog_results[1][0], self.RB.recog_results[2][0], self.RB.recog_results[3][0])
        
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def confirmRecognitionSilent(self):
        id = self.person[0]
        self.RB.confirmPersonIdentity(id=self.person[0], name=self.person[1], is_known=(not self.isUnknown)) # save the network, analysis data, csv for learning and picture of the person in the tablet
        if self.isAddPersonToDB:
            self.loadDB(self.RB.db_file)
        self.subscribeToHead()
        self.detectPeople()

    @qi.bind(returnType=qi.AnyArguments, paramsType=[]) 
    def getTime(self):
        numeric_day_name = int(time.strftime("%u")) # Numeric representation of the day of the week (1 (for Monday) through 7 (for Sunday))
        current_time = time.strftime("%T") #"%H:%M:%S" (21:34:17)
        day = time.strftime("%d")
        month = time.strftime("%B")
        year = time.strftime("%Y")
        date_today = [current_time, numeric_day_name, day, month, year]

        return date_today
    
    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])    
    def updateDB(self, csv_file, p_name):
        """This function updates the database csv file with the updated time of the person seen"""
        time_p = self.db_df.loc[self.db_df['name'] == p_name, 'times'].iloc[0]
        time_p.append(self.getTime())
        self.db_df.loc[self.db_df['name'] == p_name, 'times'].iloc[0] = time_p
        with open(csv_file, 'w') as fd:
            self.db_df.to_csv(fd, index=False, header=True)
        
    @qi.bind(returnType=qi.Bool, paramsType=[qi.String])    
    def isPersonInDB(self, p_name):
        p_name = self.changeNameLetters(p_name)
        if self.num_db > 0:
            if p_name in self.db_df.name.values:
                return True
        return False
    
    @qi.bind(returnType=qi.Void, paramsType=[qi.String])  
    def loadDB(self, csv_file):
        if os.path.isfile(csv_file):
            self.num_db = sum(1 for line in open(csv_file)) - 1
            self.db_df = pandas.read_csv(csv_file, dtype={"I": object}, converters={"times": ast.literal_eval})
        else:
            self.cleanDB()
        
    @qi.bind(returnType=qi.Void, paramsType=[qi.String])  
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
    
    @qi.bind(returnType=qi.Void, paramsType=[])
    def cleanDB(self):
        self.RB.resetFiles()
        self.s.ALFaceDetection.clearDatabase()
        self.num_db = 0

    def isNewFromName(self, thisname): # TODO delete when replaced by isPersonInDB()
        print "isNewFromName :" + str(self.db_df)
        return self.db_df.loc[self.db_df['name'] == str(thisname)].empty

    @qi.bind(returnType=qi.Void, paramsType=[])    
    def stop(self):
        "stop"
        
        self.logger.info("RecognitionModule stopped by user request.")
        self.running = False
        self.RB.saveBN()
        self.RB.saveFaceDetectionDB()
#         self.s.ALMotion.rest()
#         time.sleep(2.0)
        try:
            if self.imageShown:
                self.s.ALTabletService.hideImage()
                self.imageShown = False
            self.robot_ip = self.s.ALTabletService.robotIp()
            self.s.ALTabletService.loadUrl("http://%s/apps/recognition-service/idle.html" % self.robot_ip)
            self.s.ALTabletService.showWebview()
            self.webviewShown = True
        except Exception,e: 
            print str(e)
        
        try:
            self.faceDetected.signal.disconnect(self.idFaceDetected)
            self.peopleDetected.signal.disconnect(self.idPeopleDetected)
            self.touch_head.signal.disconnect(self.id_head)
            self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm)
            self.s.ALTabletService.onJSEvent.disconnect(self.id_tablet)
            self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm_registration)
            self.touch_bumper.signal.disconnect(self.id_bumper)
        except:
            pass
        self.unsubscribeFacePeople()
#         self.s._ALAutonomousTablet.setMode("DisplayAppScreen")
        self.qiapp.stop()
        
    @qi.nobind
    def on_stop(self):
        "Cleanup"
        self.stop()
        self.logger.info("RecognitionModule finished.")

    def run(self):
        self.onFaceDetected()

    @qi.bind(returnType=None, paramsType=(qi.List(qi.String), qi.List(qi.Float), qi.Float, qi.Float, qi.String, qi.Float, qi.Float, qi.Float, qi.List(qi.String)))
    def set_face_recog_results(self, face_names, face_confidences, recognition_accuracy, age, age_confidence, gender, gender_confidence, height, height_confidence, timestamp):
        assert len(face_names) == len(face_confidences)
        face_data = [recognition_accuracy, [ [name, conf] for name, conf in zip(face_names, face_confidences)]]
        gender_data = [gender, gender_confidence]
        age_data = [long(age), age_confidence]
        height_data = [height, height_confidence]
        self.RB.set_face_recog_results([face_data, gender_data, age_data, height_data, timestamp], face_names[0])


if __name__ == "__main__":
    stk.runner.run_service(RecognitionModule)



