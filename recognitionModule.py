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

import sys
import logging 
import time
import functools

import pandas
import ast

import RecognitionMemory

from datetime import datetime
from numpy.ma.core import ids
import os.path

def generate_name(max):
    r = range(max)
    for i in r:
        yield "person" + str(i)
        yield "person" + str(i)

@RecognitionMemory.for_all_methods(RecognitionMemory.print_function_name)
@qi.multiThreaded()
class RecognitionModule(object):
   
    APP_ID = "com.RecognitionModule"
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
        self.name_generator = generate_name(99)
        
        self.r_ip = "127.0.0.1" #NOTE: This is not to be changed if the code is running locally on the robot. If using remotely, set this to correct value.
        
        self.isRegistered = True
        self.isMemoryRobot = True
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
        
        self.isRegisteringPerson = False
        self.isRecognitionOn = False
        self.identity_est = ""
        self.isAskedForReposition = False
        self.isNameAsked = False
        self.isRegistrationAsked = False
        self.isInputAsked = False
        self.isTabletInteraction = False # True if the interaction is from tablet, False otherwise
        
    @qi.bind(returnType=qi.Void, paramsType=[])
    def blink(self):
        rDuration = 0.05
        self.leds = self.s.ALLeds
        self.leds.fadeRGB( "FaceLed0", 0x000000, rDuration )
        self.leds.fadeRGB( "FaceLed1", 0x000000, rDuration )
        self.leds.fadeRGB( "FaceLed2", 0xffffff, rDuration )
        self.leds.fadeRGB( "FaceLed3", 0x000000, rDuration )
        self.leds.fadeRGB( "FaceLed4", 0x000000, rDuration )
        self.leds.fadeRGB( "FaceLed5", 0x000000, rDuration )
        self.leds.fadeRGB( "FaceLed6", 0xffffff, rDuration )
        self.leds.fadeRGB( "FaceLed7", 0x000000, rDuration )
        time.sleep( 0.1 )
        self.leds.fadeRGB( "FaceLeds", 0xffffff, rDuration )
    
    @qi.bind(returnType=qi.Void, paramsType=[])    
    def startBlinking(self):
        if not self.s.ALAutonomousBlinking.isEnabled():
            self.s.ALAutonomousBlinking.setEnabled(True)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def stopBlinking(self):
        if self.s.ALAutonomousBlinking.isEnabled():
            self.s.ALAutonomousBlinking.setEnabled(False)
    
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onTouchedHand(self, strVarName, value):
        """ This will be called each time a touch in left hand
        is detected.
        """
        print "Left hand touched. Initializing system."
        
        self.touch_hand.signal.disconnect(self.id_hand)
     
        self.initSystem()

    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToHead(self):
        if self.touch_head is None:
            self.headTouched = False
            self.touch_head = self.s.ALMemory.subscriber(self.touch_head_event)
            self.id_head = self.touch_head.signal.connect(functools.partial(self.onTouchedHead, self.touch_head_event))
                
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onTouchedHead(self, strVarName, value):
        """ This will be called each time a touch head
        is detected.
        """
        print "Head touched. Breaking out of ignore mode for recognition."
        if self.touch_head is not None and not self.isRecognitionOn:
            self.touch_head.signal.disconnect(self.id_head)
            self.touch_head = None
            self.id_head = -1
            self.s.ALPeoplePerception.resetPopulation()
            self.headTouched = True

    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToLeftBumper(self):
        if self.touch_bumper is None:
            self.bumperPressed = False
            self.touch_bumper = self.s.ALMemory.subscriber(self.touch_bumper_event)
            self.id_bumper = self.touch_bumper.signal.connect(functools.partial(self.onTouchedLeftBumper, self.touch_bumper_event))
                
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onTouchedLeftBumper(self, strVarName, value):
        """ This will be called each time a touch left bumper
        is detected.
        """
        print "Bumper touched. Going back to idle mode."
        if self.touch_bumper is not None and self.isRecognitionOn:
            self.touch_bumper.signal.disconnect(self.id_bumper)
            self.touch_bumper = None
            self.id_bumper = -1
            if self.isNameAsked:
                try:
                    if self.webviewShown:
                        self.s.ALTabletService.hideWebview()
                        self.webviewShown = False
                    self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm)
                    self.isNameAsked = False
                except Exception,e: 
                    print str(e)
            elif self.isInputAsked:
                try:
                    if self.webviewShown:
                        self.s.ALTabletService.hideWebview()
                        self.webviewShown = False
                    self.s.ALTabletService.onJSEvent.disconnect(self.id_tablet)
                    self.isInputAsked = False
                except Exception,e: 
                    print str(e)
            elif self.isRegistrationAsked:
                try:
                    if self.webviewShown:
                        self.s.ALTabletService.hideWebview()
                        self.webviewShown = False
                    self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm_registration)
                    self.isRegistrationAsked = False
                except Exception,e: 
                    print str(e)
            self.s.ALPeoplePerception.resetPopulation()
            self.subscribeToHead()
            self.detectPeople()
            
    @qi.bind(returnType=qi.Void, paramsType=[])  
    def subscribeFacePeople(self):
        try:
            self.s.ALFaceDetection.subscribe("Test_Face", 500, 0.0)
            self.s.ALPeoplePerception.subscribe("Test_People")
        except:
            pass
        
    @qi.bind(returnType=qi.Void, paramsType=[])  
    def unsubscribeFacePeople(self):      
        "unsubscribe from the services"
        try:
            self.s.ALFaceDetection.unsubscribe("Test_Face")
            self.s.ALPeoplePerception.unsubscribe("Test_People")
        except:
            pass
                
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def initSystem(self):
        image_path = "/home/nao/.local/share/PackageManager/apps/recognition-service/html/temp.jpg"
        
        self.RB = RecognitionMemory.RecogniserBN()

        self.isMemoryRobot = True
        self.isDBinCSV = True
        self.isMultipleRecognitions = False
        self.numMultRecognitions = 3
        
        self.RB.setFilePaths(self.recog_folder)
        self.RB.connectToRobot(self.r_ip, port = 9559, useSpanish = False, isImageFromTablet = True, isMemoryOnRobot = True, imagePath = image_path)
        self.s.RecognitionService.setHeadAngles(-10.0)
        self.RB.setSessionConstant(isMemoryRobot = self.isMemoryRobot, isDBinCSV = self.isDBinCSV, isMultipleRecognitions = self.isMultipleRecognitions, defNumMultRecog = self.numMultRecognitions)

        if os.path.isfile(self.recog_folder + self.face_db):
            self.RB.useFaceDetectionDB("faceDB")
        else:
            self.cleanDB()
            self.RB.resetFaceDetectionDB()
            self.RB.setFaceDetectionDB("faceDB")
           
        # NOTE: Uncomment this to clean the files and face recognition database
        # self.cleanDB()

        # NOTE: Uncomment this to revert the files to the previous recognition state.
        # self.RB.revertToLastSaved(isRobot=True)

        self.loadDB(self.RB.db_file)
        try:
            self.robot_ip = self.s.ALTabletService.robotIp()
            self.s.ALTabletService.hideWebview()
            self.webviewShown = False
            self.id_tablet = None
            self.id_confirm = None
            self.s.ALTabletService.cleanWebview()
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
    
        self.running = True
        self.timer = 60*30
    
        self.web_link = "http://%s/apps/recognition-service/" % self.robot_ip
     
        if not self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setEnabled(True)
            
        self.s.ALAutonomousLife.setRobotOffsetFromFloor(0.1)
                               
        self.isRegisteringPerson = False
            
        self.unsubscribeFacePeople()
        self.subscribeFacePeople()
        self.startBlinking()
    
        self.subscribeToHead()
            
        self.detectPeople()
        
    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToPeopleDetected(self):
        """Modified version of RecognitionService subscribeToPeopleDetected,i.e. there is no timer in people detect"""
        self.heightPerson = None
        self.start_people_detect_time = time.time()
#         self.s.ALPeoplePerception.resetPopulation()
        self.peopleDetected = self.s.ALMemory.subscriber(self.peopleDetectedEvent)
        self.idPeopleDetected = self.peopleDetected.signal.connect(functools.partial(self.onPeopleDetected, self.peopleDetectedEvent))
#         print "waiting for people"
            
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onPeopleDetected(self, strVarName, value):
        """Modified version of RecognitionService onPeopleDetected,i.e. there is no timer in people detect"""
        self.peopleDetected.signal.disconnect(self.idPeopleDetected)
        self.peopleDetected = None
        self.idPeopleDetected = -1
        
        try:
#             print "found people"
            if self.headTouched or not self.identity_est or value[0] != self.person_id or (value[0] == self.person_id and self.s.ALMemory.getData("PeoplePerception/Person/"+str(value[0])+"/PresentSince") > self.timer):
                self.person_id = value[0]
                height_id = "PeoplePerception/Person/"+str(value[0])+"/RealHeight"
                height_taken = self.s.ALMemory.getData(height_id)*100
                self.s.RecognitionService.setHeightOfPerson(height_taken, True)
                print "time taken for people detect:" + str(time.time() - self.start_people_detect_time)
                self.subscribeToFaceDetected()
            else:
                self.person_id = None
                self.subscribeToPeopleDetected()
        except:
            self.person_id = None
            self.subscribeToPeopleDetected()


    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToFaceDetected(self):
        """Modified version of RecognitionService subscribeToFaceDetected,i.e. picture is taken after a face is found, and face results are to be found from that picture"""
        self.start_face_detect_time = time.time()
        if self.faceDetected is not None and self.idFaceDetected != -1:
            self.faceDetected.signal.disconnect(self.idFaceDetected)
        self.faceDetected = None
        self.idFaceDetected = -1
        self.faceDetected = self.s.ALMemory.subscriber(self.faceDetectedEvent)
        self.idFaceDetected = self.faceDetected.signal.connect(functools.partial(self.onFaceDetected, self.faceDetectedEvent))
#         print "waiting for face"
                
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onFaceDetected(self, strVarName, value):
        """Modified version of RecognitionService onFaceDetected,i.e. picture is taken after a face is found, and face results are to be found from that picture"""
#         print "found face"
        if self.faceDetected is None or self.idFaceDetected == -1:
            self.faceDetected = None
            self.idFaceDetected = -1
            self.subscribeToPeopleDetected()
        elif time.time() - self.start_face_detect_time > self.faceDetectTimer: #seconds
            self.faceDetected.signal.disconnect(self.idFaceDetected)
            self.faceDetected = None
            self.idFaceDetected = -1
            self.subscribeToPeopleDetected()
        else:
            self.faceDetected.signal.disconnect(self.idFaceDetected)
            self.faceDetected = None
            self.idFaceDetected = -1

            if value and value[0][3][0][1] > 0.0:
                print "time taken to find height and face" + str(time.time() - self.start_people_detect_time)
                if self.isRegisteringPerson:
                    if self.is_camera_shooting:
                        self.takePicture()
                        self.showPicture()
                        if self.RB.isMultipleRecognitions:
                            self.s.RecognitionService.setImagePathMult(0)

                    self.isRegisteringPerson = False
                    self.isAddPersonToDB= True
                    self.isRegistered = False
                    self.RB.setPersonToAdd(self.person)
                    if self.isTabletInteraction:
                        self.confirmRecognition()
                    else:
                        self.confirmRecognitionSilent()
                else:
                    if self.is_camera_shooting:
                        self.takePicture()
                    self.recog_start_time = time.time()
                    if self.isTabletInteraction:
                       self.recognise()
                    else:
                        self.recogniseSilent()
            else:
                if time.time() - self.start_face_detect_time <= self.faceDetectTimer:
                    self.subscribeToFaceDetected()
                else:
                    self.subscribeToPeopleDetected()
    #                 print "timer exceeded for face. Look for person again"
            
    @qi.bind(returnType=qi.Void, paramsType=[])
    def detectPeople(self):
        self.isRecognitionOn = False
#         if not self.s.ALBasicAwareness.isEnabled():
#             self.s.ALBasicAwareness.setEnabled(True)
        if self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setEnabled(False)
            self.s.RecognitionService.setHeadAngles(-33)
#         self.s.ALTabletService.cleanWebview()
        try:
            if not self.idleShown:
                self.s.ALTabletService.cleanWebview()
                self.s.ALTabletService.loadUrl("http://%s/apps/recognition-service/idle.html" % self.robot_ip)
                self.s.ALTabletService.showWebview()
                self.webviewShown = True
                self.idleShown = True
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
        
        self.subscribeToPeopleDetected()
        
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def recognise(self):
        self.isRegistered = True
        self.isAddPersonToDB = False
        self.isConfirmRegistration = False
        self.isRegisteringPerson = False
        self.RB.setSessionVar(isRegistered = self.isRegistered, isAddPersonToDB = self.isAddPersonToDB)    
        
        self.identity_est = self.RB.recognise_mem() # get the estimated identity from the recognition network
        print "time for recognition:" + str(time.time()-self.recog_start_time)
        print "In recognise(), identity_est = " + str(self.identity_est)
        self.counter = 0 
        self.isRecognitionCorrect = False
        self.id_tablet = None
        self.id_confirm = None

        if self.identity_est and (self.headTouched or not (self.identity_est in self.recognised_people 
               and self.recog_start_time - self.recognised_times[self.recognised_people.index(self.identity_est)][-1] < self.timer)):
            self.isRecognitionOn = True
            self.subscribeToLeftBumper()
            if not self.s.ALBasicAwareness.isEnabled():
                self.s.ALBasicAwareness.setEnabled(True)
            if self.identity_est == self.RB.unknown_var:
                textToSay = self.RB.unknownPerson
                # get tablet input 
                self.say(textToSay)
                self.askInputJS(self.parameters[self.counter])
            else:
                identity_name = self.RB.names[self.RB.i_labels.index(self.identity_est)]
                self.getPersonFromDB(identity_name)
                identity_say = identity_name.split()
                textToSay = self.RB.askForIdentityConfirmal.replace("XX", str(identity_say[0]))

                try:
                    self.id_confirm = self.s.ALTabletService.onJSEvent.connect(self.onConfirmNameJS)
                    self.say(textToSay)
                    self.isNameAsked = True
                    self.getNameJS(identity_name)
                except Exception,e:
                    print str(e)
                    self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
                    self.stop()
                                
        else:
            if self.identity_est:
                print self.RB.names[self.RB.i_labels.index(self.identity_est)]
            else:
                print "all images are discarded"

            self.subscribeToHead()
            self.detectPeople()

    @qi.bind(returnType=qi.AnyArguments, paramsType=[]) 
    def recogniseSilent(self):
        self.isRegistered = True
        self.isAddPersonToDB = False
        self.isConfirmRegistration = False
        self.isRegisteringPerson = False
        self.RB.setSessionVar(isRegistered = self.isRegistered, isAddPersonToDB = self.isAddPersonToDB)    
        
        self.identity_est = self.RB.recognise_mem() # get the estimated identity from the recognition network
        print "time for recognition:" + str(time.time()-self.recog_start_time)
        self.counter = 0 
        self.isRecognitionCorrect = False
        self.id_tablet = None
        self.id_confirm = None
        identity_name = ""
        if self.identity_est:
            self.isRecognitionOn = True
            self.subscribeToLeftBumper()
            if not self.s.ALBasicAwareness.isEnabled():
                self.s.ALBasicAwareness.setEnabled(True)
            if self.identity_est == self.RB.unknown_var:
                self.isRegistered = False
                self.isRegisteringPerson = True
                self.addPersonUsingRecogValues(self.name_generator.next()) # generate a name
                print "isRegistered : " + str(self.isRegistered) + ", id estimated: " + self.identity_est + " id name: " + identity_name
                self.s.ALMemory.raiseEvent("RecognitionResultsWritten", [self.isRegistered, self.identity_est, identity_name])

            else:
                identity_name = self.RB.names[self.RB.i_labels.index(self.identity_est)]
                print identity_name
                self.getPersonFromDB(identity_name)
                print "isRegistered : " + str(self.isRegistered) + ", id estimated: " + self.identity_est + " id name: " + identity_name
                self.s.ALMemory.raiseEvent("RecognitionResultsWritten", [self.isRegistered, self.identity_est, identity_name])  

        else:
            print "all images are discarded"

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
    def confirmRecognition(self):
        p_id = None
        self.confirm_recognition_start = time.time()
        if self.isMemoryRobot and self.isRegistered:
            p_id = str(self.person[0])
            if self.identity_est == p_id:
                self.isRecognitionCorrect = True # True if the name is confirmed by the patient
                
        if self.isRecognitionCorrect:
            self.RB.confirmPersonIdentity() # save the network, analysis data, csv for learning and picture of the person in the tablet
        else:
            p_id = str(self.person[0])
            self.RB.confirmPersonIdentity(p_id)
        self.recog_end_time = time.time()
        print "confirmation time:" + str(self.recog_end_time- self.confirm_recognition_start) 
        if p_id not in self.recognised_people:
            self.recognised_people.append(p_id)
            self.recognised_times.append([self.recog_end_time])
        else:
            self.recognised_times[self.recognised_people.index(p_id)].append(self.recog_end_time)
        if self.isAddPersonToDB:
            self.loadDB(self.RB.db_file)
        try:
            if self.imageShown:
                time.sleep(2.0)
                self.s.ALTabletService.hideImage()
                self.imageShown = False
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
        
        self.subscribeToHead()
        self.detectPeople()
        
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def confirmRecognitionSilent(self):
        p_id = None
        self.confirm_recognition_start = time.time()
        if self.isMemoryRobot and self.isRegistered:
            p_id = str(self.person[0])
            if self.identity_est == p_id:
                self.isRecognitionCorrect = True # True if the name is confirmed by the patient
                
        if self.isRecognitionCorrect:
            self.RB.confirmPersonIdentity() # save the network, analysis data, csv for learning and picture of the person in the tablet
        else:
            p_id = str(self.person[0])
            self.RB.confirmPersonIdentity(p_id)
        self.RB.saveFaceDetectionDB()
        self.recog_end_time = time.time()
        print "confirmation time:" + str(self.recog_end_time- self.confirm_recognition_start) 
        if self.isAddPersonToDB:
            self.loadDB(self.RB.db_file)

        self.subscribeToHead()
        self.detectPeople()
                
    @qi.bind(returnType=qi.Bool, paramsType=[]) 
    def isParamCorrect(self, param, value):
        is_correct = False
        if param == "name":
            is_correct = True
        elif param == "gender":
            is_correct = True
        elif param == "age":
            try:
                birth_year = int(value)
                cur_year = int(time.strftime("%Y"))
                value = cur_year - birth_year
                if value > 100:
                    self.say("Are you really older than 100? You look much younger! Or did I get that wrong?")
                elif value == 42:
                    self.say("You might be the answer to my questions!")
                    is_correct = True
                elif value < 0:
                    self.say("Wow! Do you really come from the future? Or did I get that wrong?")
                else:
                    is_correct = True
            except:
                self.say("Could you please enter your birth year?")
        elif param == "height":
            try:
                value = int(value)
                if value < 40:
                    self.say("Could you please enter your height in centimeters?")
                elif value > 240:
                    self.say("Are you taller than the tallest man on Earth? You should apply for Guinness world records! Or did I get that wrong?")
                else:
                    is_correct = True
            except:
                self.say("Could you please enter your height in centimeters?")
        return is_correct
       
    @qi.bind(returnType=qi.AnyArguments, paramsType=[qi.String, qi.String]) 
    def setCorrectValue(self, param, value):
        if param =="name":
            value = self.changeNameLetters(value)
        elif param == "age":
            value = int(value)
        elif param == "height":
            value = int(value)
            if value < self.RB.height_min:
                value = self.RB.height_min
        return value
    
    @qi.bind(returnType=qi.Void, paramsType=[])
    def takePicture(self):
        if self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setEnabled(False)
        resolution = 2    # VGA
        camera_id = 0 # Top camera
        image_dir = "/home/nao/.local/share/PackageManager/apps/recognition-service/html/"
        file_name = "temp.jpg"
        self.photo_capture = self.s.ALPhotoCapture
        self.photo_capture.setResolution(resolution)
        self.photo_capture.setCameraID(camera_id)
        self.photo_capture.setPictureFormat("jpg")
        pic_start_time = time.time()
        if self.RB.isMultipleRecognitions:
            for num_recog in range(0, self.RB.def_num_mult_recognitions):
                temp_start = time.time()
                to_rep = str(num_recog) + ".jpg"
                file_name_new = file_name.replace(".jpg", to_rep)
                self.photo_capture.takePicture( image_dir, file_name_new )
                if num_recog == 0 and self.isRegisteringPerson:
                    self.sayNoMovement("Let me see how it looks like")
                # print time.time() - temp_start
        else:
            self.photo_capture.takePicture( image_dir, file_name )
            if self.isRegisteringPerson:
                self.sayNoMovement("Let me see how it looks like")
        print "time to take pictures: " + str(time.time() - pic_start_time)
        if not self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setEnabled(True)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def showPicture(self):
        if self.RB.isMultipleRecognitions:
            image_name = "http://%s/apps/recognition-service/temp0.jpg" % self.robot_ip
        else:
            image_name = "http://%s/apps/recognition-service/temp.jpg" % self.robot_ip
#         self.s.ALTabletService.cleanWebview()
        try:
            if self.webviewShown:
                self.s.ALTabletService.hideWebview()
                self.webviewShown = False
            self.s.ALTabletService.showImageNoCache(image_name)
            
            self.imageShown = True
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
        

    @qi.bind(returnType=qi.String, paramsType=[qi.String]) 
    def changeNameLetters(self, name):
        name = name.lower()
        name_c = name.title()
        return name_c

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def getRegistrationJS(self, person):
        """ Build registration message with parameters given from person"""

        script = """
                window.onload = function(){
                    document.getElementById('11').innerHTML = 'Name: XX';
                    document.getElementById('22').innerHTML = 'Gender: YY';
                    document.getElementById('33').innerHTML = 'Birth Year: WW';
                    document.getElementById('44').innerHTML = 'Height: ZZ';
                    var timeout_period = 90000;
                    var time_start = new Date().getTime();
                    var refreshFunc = setInterval(refresh, 1000);
                    document.getElementById('sub').onclick = function(){
                        var arr = [];
                        for(count = 1; count < 5; count++){
                            if(document.getElementById(count).checked){
                                arr.push(document.getElementById(count).value);
                            }
                        }
                        var str = arr.join(",");
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(str);
                    };
                    function refresh() {
                         if(new Date().getTime() - time_start >= timeout_period){
                            clearInterval(refreshFunc);
                            ALTabletBinding.raiseEvent('timeout');
                         }
                    } 
                };
            """
        script = script.replace("XX", person[1])
        script = script.replace("YY", person[2])
        script = script.replace("WW", str(person[3]))
        script = script.replace("ZZ", str(person[4]))
#         script = script.replace("timeoutPeriod", str(self.timeoutPeriod))
        self.s.ALTabletService.loadUrl(self.web_link + "confirmRegistration.html")
        self.s.ALTabletService.executeJS(script)
        self.s.ALTabletService.showWebview()
        self.webviewShown = True
        self.idleShown = False
   
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def getInputJS(self, param):
        if param == "gender":
            script = """
                window.onload = function(){
                    var timeout_period =90000;
                    var time_start = new Date().getTime();
                    var refreshFunc = setInterval(refresh, 1000);
                    document.getElementById('1').onclick = function() {
                        var param_value = document.getElementById('1').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    document.getElementById('2').onclick = function() {
                        var param_value = document.getElementById('2').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    function refresh() {
                         if(new Date().getTime() - time_start >= timeout_period){
                            clearInterval(refreshFunc);
                            ALTabletBinding.raiseEvent('timeout');
                         }
                    } 
                };
            """
        elif param == "height":
            script = """
                window.onload = function(){
                    var timeout_period = 180000;
                    var time_start = new Date().getTime();
                    var refreshFunc = setInterval(refresh, 1000);
                    document.getElementById('sub').onclick = function() {
                        var param_value = document.getElementById('param').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    function refresh() {
                         if(new Date().getTime() - time_start >= timeout_period){
                            clearInterval(refreshFunc);
                            ALTabletBinding.raiseEvent('timeout');
                         }
                    } 
                };
            """
        else:
            script = """
                window.onload = function(){
                    var timeout_period = 90000;
                    var time_start = new Date().getTime();
                    var refreshFunc = setInterval(refresh, 1000);
                    document.getElementById('sub').onclick = function() {
                        var param_value = document.getElementById('param').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    function refresh() {
                         if(new Date().getTime() - time_start >= timeout_period){
                            clearInterval(refreshFunc);
                            ALTabletBinding.raiseEvent('timeout');
                         }
                    } 
                };
            """
#         script = script.replace("timeoutPeriod", str(self.timeoutPeriod))
        self.s.ALTabletService.loadUrl(self.web_link + param + ".html")
        self.s.ALTabletService.executeJS(script)                 
        self.s.ALTabletService.showWebview()
        self.webviewShown = True
        self.idleShown = False
        
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def getNameJS(self, identity_name):
        script = """
                window.onload = function(){
                    var timeout_period = 90000;
                    document.getElementById('prompt-message').innerHTML = 'Hello XX! Is that you?';
                    var time_start = new Date().getTime();
                    var refreshFunc = setInterval(refreshF, 1000);
                    document.getElementById('1').onclick = function() {
                        var param_value = document.getElementById('1').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    document.getElementById('2').onclick = function() {
                        var param_value = document.getElementById('2').value;
                        clearInterval(refreshFunc);
                        ALTabletBinding.raiseEvent(param_value);
                    };
                    function refreshF() {
                        if(new Date().getTime() - time_start >= timeout_period){
                            clearInterval(refreshFunc);
                            ALTabletBinding.raiseEvent('timeout');
                        }
                    }
                };
            """
        script = script.replace("XX", identity_name)
        # script = script.replace("timeoutPeriod", str(self.timeoutPeriod))
        self.s.ALTabletService.loadUrl(self.web_link + "confirmName.html") 
        self.s.ALTabletService.executeJS(script)              
        self.s.ALTabletService.showWebview()
        self.webviewShown = True
        self.idleShown = False

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def askRegistrationJS(self):
#         self.s.ALTabletService.cleanWebview()
        try:
            self.id_confirm_registration = self.s.ALTabletService.onJSEvent.connect(self.onConfirmRegistrationJS)
            self.getRegistrationJS(self.person)
            self.isRegistrationAsked = True
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
                
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def askInputJS(self, param):
#         self.s.ALTabletService.cleanWebview()
        try:
            self.id_tablet = self.s.ALTabletService.onJSEvent.connect(self.onInputJS)
            self.getInputJS(param)
            self.isInputAsked = True
        except Exception,e:
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
            
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onInputJS(self, value):
        """ This will be called each time a tablet input is required."""
        # Unsubscribe to the event when talking,
        # to avoid repetitions
        try:
            if self.webviewShown:
                self.s.ALTabletService.hideWebview()
                self.webviewShown = False
            self.s.ALTabletService.onJSEvent.disconnect(self.id_tablet)
            self.isInputAsked = False
        except Exception,e: 
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()    
        if not value:
            self.say("I am sorry, I didn't quite get that. Could you enter your " + self.parameters[self.counter] + " again please?")
            self.askInputJS(self.parameters[self.counter])
        elif value == "timeout":
            print "time out"
            self.subscribeToHead()
            self.detectPeople()
        elif not self.isParamCorrect(self.parameters[self.counter], value):
            self.askInputJS(self.parameters[self.counter])
        elif self.counter == 0 and self.isPersonInDB(value): #if the name is in database
            self.isRegistered = True
            value = self.changeNameLetters(value)
            self.getPersonFromDB(value)
            self.confirmRecognition()
        else:
            self.person[self.counter+1] = self.setCorrectValue(self.parameters[self.counter], value)
            if self.isConfirmRegistration:
                self.confirm_counter.pop(0)
                if self.confirm_counter:
                    self.counter = self.confirm_counter[0]
                    if self.counter == 1:
                        self.say("What is your gender?")
                    elif self.counter == 2:
                        self.say("In what year were you born?")
                    elif self.counter == 3:
                        self.say("How tall are you in centimeters?")
                    else:
                        self.say("Your " + self.parameters[self.counter] + " please?")
                    # Reconnect again to the event                 
                    self.askInputJS(self.parameters[self.counter])    
                else:
                    # confirm screen
                    self.say("Is there anything else that is incorrect?")
                    self.askRegistrationJS()         
            else:
                if self.counter == 0:
                    self.person[self.counter] = str(self.num_db + 1)
                self.counter += 1
                if self.counter == len(self.parameters):
                    # confirm screen
                    self.say("Okay! Did I get anything wrong? If I didn't, press the submit button right ahead!")
                    self.askRegistrationJS()
                else:
                    if self.counter == 1:
                        self.say("Oh we haven't met before! I would like to know more about you! For example, most people think I am a female, but in Japan they see me as a male! What is \\emph=2\\ your gender?")
                    elif self.counter == 2:
                        self.say("You look very young! When were you born?")
                    elif self.counter == 3:
                        self.say("How tall are you in centimeters? I am 121, not the right size to play basketball!")
                    else:
                        self.say("Your " + self.parameters[self.counter] + " please?")
                                    
                    # Reconnect again to the event                 
                    self.askInputJS(self.parameters[self.counter])    

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onConfirmRegistrationJS(self, value):
        """ This will be called each time a tablet input is required."""
        # Unsubscribe to the event when talking,
        # to avoid repetitions
        try:
            if self.webviewShown:
                self.s.ALTabletService.hideWebview()
                self.webviewShown = False
            self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm_registration)
            self.isRegistrationAsked = False
        except Exception,e: 
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
#         print value        
        if not value:
            self.person[-1] = [self.getTime()]
            self.RB.saveImageAfterRecognition(False, str(self.person[0])) # save the previous images before replacing them
#             print self.person
            self.say("Great! Now I will take a picture, so I can remember you better the next time I see you!")
            self.isRegisteringPerson = True
            self.isAskedForReposition = False
            self.subscribeToFaceDetected()
        elif value == "timeout":
            print "time out"
            self.subscribeToHead()
            self.detectPeople()    
        else:
            self.isConfirmRegistration = True
            self.say("Let's do this again then!")
            self.confirm_counter = map(int, value.split(','))
#             print self.confirm_counter
            self.counter = self.confirm_counter[0]
            if self.counter == 0:
                self.say("What is your name?")
            elif self.counter == 1:
                self.say("What is your gender?")
            elif self.counter == 2:
                self.say("In what year were you born?")
            elif self.counter == 3:
                self.say("How tall are you in centimeters?")
            else:
                self.say("Your " + self.parameters[self.counter] + " please?")
            self.askInputJS(self.parameters[self.counter])
            
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onConfirmNameJS(self, value):
        """ This will be called each time a tablet input is required."""
        # Unsubscribe to the event when talking,
        # to avoid repetitions
        try:
            if self.webviewShown:
                self.s.ALTabletService.hideWebview()
                self.webviewShown = False
            self.s.ALTabletService.onJSEvent.disconnect(self.id_confirm)
            self.isNameAsked = False
        except Exception,e: 
            print str(e)
            self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
            self.stop()
            
        print "confirm value:" + value
        if value == "true":
            self.confirmRecognition()
        elif value == "false":
            self.say("No? I am sorry for that. Could you enter your name please?")
#             self.s.ALTabletService.cleanWebview()
            try:
                self.s.ALTabletService.loadUrl(self.web_link + self.parameters[0] + ".html")
                self.s.ALTabletService.showWebview()
                self.webviewShown = True   
                self.id_tablet = self.s.ALTabletService.onJSEvent.connect(self.onInputJS)
                self.isInputAsked = True
                self.getInputJS(self.parameters[self.counter])  
            except Exception,e:
                print str(e)
                self.say("There seems to be an error with my tablet. Could you notify the experimenter please?")
                self.stop()
                
        elif value == "timeout":
            print "time out"
            self.subscribeToHead()
            self.detectPeople()
            
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def say(self, sentence):
        self.configuration = {"bodyLanguageMode":"contextual"}
        self.s.ALTextToSpeech.setVolume(0.85)
        self.s.ALTextToSpeech.setParameter("speed", 95)
        self.s.ALAnimatedSpeech.say(sentence,self.configuration)
    
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def sayNoMovement(self, sentence):
        self.configuration = {"bodyLanguageMode":"disabled"}
        self.s.ALTextToSpeech.setVolume(0.85)
        self.s.ALTextToSpeech.setParameter("speed", 95)
        self.s.ALAnimatedSpeech.say(sentence,self.configuration)   
            
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

if __name__ == "__main__":
    stk.runner.run_service(RecognitionModule)



