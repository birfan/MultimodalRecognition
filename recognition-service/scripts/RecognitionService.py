# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright 2017 Bahar Irfan                                                                            #
#                                                                                                        #
#  RecogniserService is a script for obtaining multi-modal information from the user: face similarity    #
#  scores, gender, age, height estimations of the user through NAOqi modules (ALFaceDetection and        #
#  ALPeopleDetection, NAOqi 2.4 and 2.5 - works on both NAO and Pepper). This information is used by     #
#  RecogniserMemory, amulti-modal incremental Bayesian network (MMIBN) with option for online learning   #
#  (evidence based updating of likelihoods) for reliable recognition in open-world identification.       #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #                 
#  RecognitionService, RecognitionMemory and each script in this project is under the GNU General Public #
#  License.                                                                                              #
#========================================================================================================#

import qi
import stk.runner
import stk.events
import stk.services
import stk.logging
import time
import functools
import almath
import time
from numpy.ma.core import ids

@qi.multiThreaded()
class RecognitionService(object):
   
    APP_ID = "com.RecognitionService"
    def __init__(self, qiapp):
        # generic activity boilerplate
        self.qiapp = qiapp
        self.versionV = "1.0"
        self.events = stk.events.EventHelper(qiapp.session) 
        self.s = stk.services.ServiceCache(qiapp.session) 
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)
        self.robot_name = "nao"
        self.isImageFromTablet = True
        self.image = "/home/nao/dev/images/nao_image.jpg" #TODO: set this!
        self.image_base = self.image
        self.robotOffsetFromFloorInMeters = 0.59 #TODO: set this to the correct value!
        self.prob_threshold = 1.0e-75
        self.subscribed = False
        self.useSpanish = False
        self.timePerson = None

    @qi.bind(returnType=qi.Void, paramsType=[qi.Bool, qi.Bool, qi.String])  
    def initSystem(self, useSpanish, isImageFromTablet, imagePath = ""):
        "Initializes the system"        
        # load custom (Alex Mazel's) library for recognising and learning from file through the tablet only if Naoqi > 2.1
        robot_version = self.s.ALMemory.version()
        if isImageFromTablet:
            if robot_version[:3] == "2.1":
                self.logger.info("Recognizing from an image is only supported in Naoqi 2.4. Please update your system.")
                self.stop()
        
        autoState = self.s.ALAutonomousLife.getState()
        if autoState != "disabled":
            self.s.ALAutonomousLife.setState("disabled")
        self.s.ALMotion.wakeUp()
        self.s.ALMotion.setBreathConfig([["Bpm", 6], ["Amplitude", 0.9]])
        self.s.ALMotion.setBreathEnabled("Arms", True)
        self.s.ALMotion.setBreathEnabled("Legs", False)
        self.s.ALAutonomousLife.setRobotOffsetFromFloor(self.robotOffsetFromFloorInMeters)
        self.s.ALAnimatedSpeech.setBodyLanguageMode(2) # contextual
        
        self.useSpanish = useSpanish
        if self.useSpanish:
            self.s.ALTextToSpeech.setLanguage("Spanish")
        else:
            self.s.ALTextToSpeech.setLanguage("English")
        
        self.isImageFromTablet = isImageFromTablet
        if imagePath:
            self.image = imagePath
            self.image_base = imagePath
        self.heightPerson = None
        self.faceResults = None

        self.unsubscribeFacePeople()
        self.subscribeFacePeople()
        
        # People detected event initialisation
        self.peopleDetectedEvent = "PeoplePerception/PeopleList"
        self.peopleDetected = None
        self.idPeopleDetected = -1
        
        self.heightPerson = None
        self.heightConf = 0
        self.idPerson = None
        self.peopleDetectTimer = 10.0
                
        # Face detected event initialisation
        self.faceDetectedEvent = "FaceDetection/FaceDetected"
        self.faceDetected = None
        self.idFaceDetected = -1
        
        self.faceResults = None
        self.faceDetectTimer = 5.0
        
    @qi.bind(returnType=qi.Void, paramsType=[qi.Float]) 
    def setHeadAngles(self, angle):
        # Simple command for the HeadYaw joint at 10% max speed
        names            = ["HeadPitch","HeadYaw"]
        angles           = [angle*almath.TO_RAD, -3.0*almath.TO_RAD]
        fractionMaxSpeed = 0.1
        self.s.ALMotion.setAngles(names,angles,fractionMaxSpeed)
    
    @qi.bind(returnType=qi.AnyArguments, paramsType=[]) 
    def getDateTime(self):
        day = time.strftime("%d")
        month = time.strftime("%B")
        year = time.strftime("%Y")
        numericDayName = time.strftime("%u") # Numeric representation of the day of the week (1 (for Monday) through 7 (for Sunday))
        currentTime = time.strftime("%T") #"%H:%M:%S" (21:34:17)
        dateToday = [currentTime, numericDayName, day, month, year]
        return dateToday
    
    @qi.bind(returnType=qi.Void, paramsType=[])
    def startAwareness(self):
        self.s.ALMotion.setBreathEnabled("Legs", True)
        if not self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("People", True)
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("Sound", False)
            self.s.ALBasicAwareness.setEngagementMode("FullyEngaged")
            self.s.ALBasicAwareness.setTrackingMode("Head")
            self.s.ALBasicAwareness.setEnabled(True)
            # time.sleep(1.0)
            
    @qi.bind(returnType=qi.Void, paramsType=qi.AnyArguments)
    def engagePerson(self, personID=None):
        if not self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("People", True)
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("Sound", False)
            self.s.ALBasicAwareness.setEngagementMode("FullyEngaged")
            self.s.ALBasicAwareness.setTrackingMode("Head")
            self.s.ALBasicAwareness.setEnabled(True)
            if not self.s.ALAutonomousBlinking.isEnabled():
                self.s.ALAutonomousBlinking.setEnabled(True)
            if personID is not None:
                self.s.ALBasicAwareness.engagePerson(personID)
        elif self.s.ALBasicAwareness.getEngagementMode() == "Unengaged":
            self.s.ALBasicAwareness.setEngagementMode("FullyEngaged")
            if personID is not None:
                self.s.ALBasicAwareness.engagePerson(personID)
    
    @qi.bind(returnType=qi.Void, paramsType=[])
    def disengagePerson(self):
        if self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setEnabled(False)
            if self.s.ALAutonomousBlinking.isEnabled():
                self.s.ALAutonomousBlinking.setEnabled(False)
                
    @qi.bind(returnType=qi.Void, paramsType=[])
    def unengageRobot(self):
        if not self.s.ALBasicAwareness.isEnabled():
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("People", True)
            self.s.ALBasicAwareness.setStimulusDetectionEnabled("Sound", False)
            self.s.ALBasicAwareness.setEngagementMode("Unengaged")
            self.s.ALBasicAwareness.setTrackingMode("Head")
            self.s.ALBasicAwareness.setEnabled(True)
            if not self.s.ALAutonomousBlinking.isEnabled():
                self.s.ALAutonomousBlinking.setEnabled(True)
        elif self.s.ALBasicAwareness.getEngagementMode() == "FullyEngaged":
            self.s.ALBasicAwareness.setEngagementMode("Unengaged")
    
    @qi.bind(returnType=qi.Void, paramsType=[qi.Int32])
    def rotateHead(self, difAngle):
        names            = ["HeadYaw"]
        angles           = difAngle*almath.TO_RAD
        fractionMaxSpeed = 0.1
        self.s.ALMotion.changeAngles(names,angles,fractionMaxSpeed)  
    
    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def setImagePath(self, image_path):
        self.image = image_path
        
    @qi.bind(returnType=qi.Void, paramsType=[qi.Float, qi.Bool])  
    def setHeightOfPerson(self, height, isHeightKnown):     
        if isHeightKnown:
            self.heightPerson = height
            self.heightConf = 0.08
        else:
            self.heightPerson = 165.0
            self.heightConf = 0
            
    @qi.bind(returnType=qi.Void, paramsType=[qi.AnyArguments])  
    def setTimeOfPerson(self, timePerson):     
        self.timePerson = timePerson
        
    @qi.bind(returnType=qi.Void, paramsType=[])  
    def setFaceResults(self, face):
        self.faceResults = face

    @qi.bind(returnType=qi.Void, paramsType=[])  
    def subscribeFacePeople(self):
        try:
            self.s.ALFaceDetection.subscribe("Test_Face_RS", 500, 0.0)
            self.s.ALPeoplePerception.subscribe("Test_People_RS", 500, 0.0)
        except:
            pass
        
    @qi.bind(returnType=qi.Void, paramsType=[])  
    def unsubscribeFacePeople(self):      
        "unsubscribe from the services"
        try:
            self.s.ALFaceDetection.unsubscribe("Test_Face_RS")
            self.s.ALPeoplePerception.unsubscribe("Test_People_RS")
        except:
            pass
                            
    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToPeopleDetected(self):
        
        self.heightPerson = None
        self.idPerson = None
        self.start_people_detect_time = time.time()
        self.s.ALPeoplePerception.resetPopulation()
        self.peopleDetected = self.s.ALMemory.subscriber(self.peopleDetectedEvent)
        self.idPeopleDetected = self.peopleDetected.signal.connect(functools.partial(self.onPeopleDetected, self.peopleDetectedEvent))
        
    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onPeopleDetected(self, strVarName, value):
        
        self.peopleDetected.signal.disconnect(self.idPeopleDetected)
        self.peopleDetected = None
        self.idPeopleDetected = -1
        
        try:
            self.idPerson = value[0]
            height_id = "PeoplePerception/Person/"+str(value[0])+"/RealHeight"
            self.setHeightOfPerson(self.s.ALMemory.getData(height_id)*100, True)
            if self.isImageFromTablet:
                recog_results = self.recognisePerson()
            else:
                self.subscribeToFaceDetected()
        except:
            if time.time() - self.start_people_detect_time <= self.peopleDetectTimer:
                self.subscribeToPeopleDetected()
            else:
                self.setHeightOfPerson(165, False)
                if self.isImageFromTablet:
                    recog_results = self.recognisePerson()
                else:
                    self.subscribeToFaceDetected()
        
        
    @qi.bind(returnType=qi.Void, paramsType=[])    
    def subscribeToFaceDetected(self):
        
        self.faceResults = None
        self.face_detect_start_time = time.time()
        self.faceDetected = self.s.ALMemory.subscriber(self.faceDetectedEvent)
        self.idFaceDetected = self.faceDetected.signal.connect(functools.partial(self.onFaceDetected, self.faceDetectedEvent))

    @qi.bind(returnType=qi.Void, paramsType=[]) 
    def onFaceDetected(self, strVarName, value):
        
        self.faceDetected.signal.disconnect(self.idFaceDetected)
        self.faceDetected = None
        self.idFaceDetected = -1
        
        if value and value[0][3][0][1] > 0.0:
            self.setFaceResults(value)
            recog_results = self.recognisePerson()
#             print "time taken to find height and face" + str(time.time() - self.start_people_detect_time) 
        else:
            if time.time() - self.face_detect_start_time <= self.faceDetectTimer:
                self.subscribeToFaceDetected()
            else:
                self.subscribeToPeopleDetected()
#                 print "timer exceeded for face. Look for person again"

    @qi.bind(returnType=qi.AnyArguments, paramsType=[])
    def recognisePerson(self):

        """ This will be called when a person
        is detected (Takes longer than a face to be detected but it is necessary for PeoplePerception module).
        """
        recog_results = []
        heightConf = self.heightConf

        height_o = self.heightPerson
        if height_o < 150 and heightConf > 0:
            # is sitting
            height_o += 26
        height = float("{0:.1f}".format(height_o))
        heightWithConfidence = [height, heightConf]
        if self.timePerson is None:
            dateToday = self.getDateTime()
        else:
            dateToday = self.timePerson
        self.timePerson = None        
        if self.isImageFromTablet:
            results = self.s.ALFaceDetection.recognizeFromFile(self.image, 0)
        else:
            results = self.faceResults
        if not results:
            self.s.ALMemory.raiseEvent("RecognitionResultsUpdated", recog_results)
            return recog_results # []
        else:
            if results[0][5]:
                faceWithConfidence = results[0][5] # faces with confidences
                for counter in range(0, len(faceWithConfidence)):
                    faceWithConfidence[counter][1] = float("{0:.3f}".format(faceWithConfidence[counter][1]))
            else:
                faceWithConfidence = []
            faceAccuracy = float("{0:.3f}".format(results[0][0][6])) # accuracy of the face detection algorithm
            faceWithConfidence = [faceAccuracy, faceWithConfidence]
        
            if results[0][3][1][0] == 0:
                gender = "Female"
            else:
                gender = "Male"
                
            genderConf = results[0][3][1][1] # gender confidence
            genderProb = float("{0:.3f}".format(self.getGenderProbability(genderConf)))
            genderWithConfidence = [gender, genderProb]
            
            age = results[0][3][0][0]
            ageConf = float("{0:.3f}".format(results[0][3][0][1])) # age confidence
            ageWithConfidence = [age, ageConf]
        recog_results = [faceWithConfidence, genderWithConfidence, ageWithConfidence, heightWithConfidence, dateToday]
        self.s.ALMemory.raiseEvent("RecognitionResultsUpdated", recog_results)
        return recog_results
    
    @qi.bind(returnType=qi.Float, paramsType=[qi.Float])  
    def getGenderProbability(self, genderConf):
        # difference between the male and female confidences is assumed to be genderConf: x - (1.0 - x) = genderConf
        return 0.5 + (genderConf/2.0)
        
        
    @qi.bind(returnType=qi.Void, paramsType=[])  
    def getPersonProperties(self):
        self.robotOffsetFromFloorInMeters = 0
        self.s.ALAutonomousLife.setRobotOffsetFromFloor(self.robotOffsetFromFloorInMeters)
        
        self.startAwareness()
        ids = []
        genderP = [0,0]
        ageP = [0,0]
        heightP = 0
        personAnalysed = False
        if self.heightPerson is None:
            self.subscribeToPeopleDetected()
        else:
            self.s.ALFaceCharacteristics.analyzeFaceCharacteristics(self.idPerson)
            genderP = self.s.ALMemory.getData("PeoplePerception/Person/"+str(self.idPerson)+"/GenderProperties")
            ageP = self.s.ALMemory.getData("PeoplePerception/Person/"+str(self.idPerson)+"/AgeProperties")
            heightP = float("{0:.1f}".format(self.heightPerson)) # in cm
            if heightP < 150:
                # is sitting
                heightP += 40
            personAnalysed = True
        return [personAnalysed, genderP, ageP, heightP]    
                
        
    @qi.bind(returnType=qi.Bool, paramsType=[qi.String])    
    def isPersonRegistered(self, p_name):
        registered_list = self.s.ALFaceDetection.getLearnedFacesList()
        return p_name in registered_list
 
    @qi.bind(returnType=qi.Void, paramsType=[qi.Int32])  
    def setImagePathMult(self, num_recog):
        to_rep = str(num_recog) + ".jpg"
        self.image = self.image_base.replace(".jpg", to_rep)
           
    @qi.bind(returnType=qi.Bool, paramsType=[qi.String])    
    def registerPerson(self, p_name):
        "adds the person in the db"
        # TODO: make sure this is not bothering or there is a button on the tablet
        p_name = str(p_name)
        learn_face_success = self.learnFace(p_name)
     
        if not learn_face_success:
            if self.isImageFromTablet:
                self.s.ALAnimatedSpeech.say("Hmm, there seems to be a problem with the image. Could you look at the tablet again please?")  
            else:
                self.s.ALAnimatedSpeech.say("Hmm, I couldn't get a good look at your face. Could you look at me again please?")
                face = self.s.ALMemory.getData("FaceDetected")
                while not face:
                    face = self.s.ALMemory.getData("FaceDetected")
                    time.sleep(0.1)
                    learn_face_success = self.s.ALFaceDetection.learnFace(p_name)
                    
        return learn_face_success

    @qi.bind(returnType=qi.Bool, paramsType=[qi.String])    
    def learnFace(self, p_name):
        if self.isPersonRegistered(p_name):
            learn_face_success = self.addPictureToPerson(p_name)
        else:
            if self.isImageFromTablet:
                learn_face_success = self.s.ALFaceDetection.learnFaceFromFile(p_name, self.image, 0)     
            else:
                learn_face_success = self.s.ALFaceDetection.learnFace(p_name)
        return learn_face_success
    
    @qi.bind(returnType=qi.Bool, paramsType=[qi.String, qi.Int32])    
    def registerPersonOnRobot(self, p_name, num_trials):
        "adds the person in the db"
        # TODO: make sure this is not bothering or there is a button on the tablet
        p_name = str(p_name)
            
        learn_face_success = self.learnFace(p_name)
        if not learn_face_success:
            if num_trials == 0:
                textToSay = "Hmm, I can't see your face properly. Could you reposition yourself and look at me please?" 
            elif num_trials == 1:
                textToSay = "I am still struggling to see your face. If you have anything covering it, like hair, hats or even masks, could you remove it please?"
            else:
                textToSay = "I am so sorry, but I have to keep on trying until I see your face properly."
            
            self.s.ALAnimatedSpeech.say(textToSay) 
        return learn_face_success        
            
    @qi.bind(returnType=qi.Bool, paramsType=[qi.String])    
    def addPictureToPerson(self, p_name):
        "adds a new picture for the person in the db"
        p_name = str(p_name)
        if self.isImageFromTablet:
            learn_face_success = self.s.ALFaceDetection.learnFaceFromFile(p_name, self.image, 1)
        else:
            learn_face_success = self.s.ALFaceDetection.reLearnFace(p_name)
        return learn_face_success

    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def say(self, sentence):
        self.configuration = {"bodyLanguageMode":"contextual"}
        self.s.ALTextToSpeech.setVolume(0.85)
        self.s.ALTextToSpeech.setParameter("speed", 95)
        self.s.ALAnimatedSpeech.say(sentence,self.configuration)
        
    @qi.bind(returnType=qi.Void, paramsType=[qi.String]) 
    def sayNoMovement(self, sentence):
        self.s.ALTextToSpeech.setVolume(0.85)
        self.s.ALTextToSpeech.setParameter("speed", 95)
        self.s.ALTextToSpeech.say(sentence)    

    @qi.bind(returnType=qi.Void, paramsType=[])    
    def stop(self):
        "stop"
        self.logger.info("RecognitionService stopped by user request.")
        
        try:
            self.faceDetected.signal.disconnect(self.idFaceDetected)
            self.peopleDetected.signal.disconnect(self.idPeopleDetected)
        except:
            pass
            
        self.unsubscribeFacePeople()
#         self.s.ALMotion.rest()
#         time.sleep(2.0)
        self.qiapp.stop()
        
    @qi.nobind
    def on_stop(self):
        "Cleanup"
        self.stop()
        self.logger.info("RecognitionService finished.")

if __name__ == "__main__":
    stk.runner.run_service(RecognitionService)

