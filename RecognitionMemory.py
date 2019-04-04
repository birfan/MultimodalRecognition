# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright 2017 Bahar Irfan                                                                            #
#                                                                                                        #
#  RecogniserMemory class is a multi-modal incremental Bayesian network (MMIBN) with option for          #
#  online learning (evidence based updating of likelihoods) for reliable recognition in open-world       #
#  identification. The primary biometric in our system is face recognition, which is fused with          #
#  soft biometrics, namely, gender, age, and height estimations and the time of interaction.             #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #
#  The pyAgrum library is used for implementing the Bayesian network structure:                           #
#    Gonzales, Christophe and Torti, Lionel and Wuillemin, Pierre-Henri (2017), 'aGrUM: a Graphical      #
#    Universal Model framework', Proceedings of the 30th International Conference on Industrial          #
#    Engineering, Other Applications of Applied Intelligent Systems.                                     #
#                                                                                                        #                      
#  RecognitionMemory and each script in this project is under the GNU General Public License.            #
#========================================================================================================#

import pyAgrum as gum
import numpy as np
import pandas

import math
import heapq # for finding second largest

import sys
import os
import os.path
import shutil
from distutils.dir_util import copy_tree # copy directory to another
    
from datetime import datetime
from datetime import date
import time

import csv
import ast

import json
from collections import OrderedDict

from multiprocessing.dummy import Pool as ThreadPool 

import itertools # for permutations to generate stats (not necessary for code)
import logging

import qi
import functools
import threading
import random # for making random choices for the phrases

class RecogniserBN:
    
    def __init__(self):
#         np.set_printoptions(threshold=np.nan)
        """
        NODES: 

        Identity: "1", "2", "3".., "0" (for "unknown")
        Range variable because it is easier to change the number of states of the node as the database grows
        
        Face: "1"'s face, "2"'s face, "3"'s face.., "0" (for "unknown")    
        See "Identity" variable explanation

        Gender: Labelized variable. Female (0 in Naoqi), Male (1 in Naoqi) 

        Age: Range variable in [0, 75] (because Naoqi age detection is in the range of [0,75] )
        NOTE: P(A=26|I=John) = this should be a Gaussian curve at mean 26, and stddev depends on the confidence of the age recognition algorithm
        
        Height: Range variable [50, 240]
        P(H=175|I=John) = this should be a Gaussian curve at mean 175. Stddev is dependent on the height recognition rate of the algorithm

        Location: Range variable (can be changed to labelized variable). 
        Kitchen, bedroom, living room, office (the places can change depending on the experiment)
        !!!!!NOTE_COLOMBIA: Not used as the location is the same!!!!!
        
        Time: Range variable
        Number of states of T: 7(days)*24(hours)*60(minutes)/30(period)
        
        In RecogniserBN.csv: Field 'R' is used to identify the registering. If the person is registering for the first
        time the value is 1, otherwise 0.
        """
        
        """FILES AND FOLDERS"""
        self.recog_folder = "" # folder for files, if nothing set, the files will be written in the current directory
        self.recog_file = "RecogniserBN.bif" # file for saving the bayesian network 
        self.recogniser_csv_file = "RecogniserBN.csv" # file for recording the evidence from the modalities (for registered users) and the actual identity of the user
        self.initial_recognition_file = "InitialRecognition.csv" # file for recording the initial evidence (before the user registers if not registered or the same evidence as RecogniserBN.csv) from the modalities and the estimated identity of the user
        self.analysis_dir = "Analysis/"
        self.analysis_file = self.analysis_dir + "Analysis.json" # file for recording all the values of the recognition (not used, slows the system)
        self.comparison_file = self.analysis_dir + "Comparison.csv" # file for comparing the recognition of face and network, and the quality of estimation
        self.db_file = "db.csv" # database file
        self.stats_file = "stats.csv" # statistics file
        self.conf_matrix_file = "confusionMatrix.csv" # confusion matrix
        self.image_save_dir = "images/" # images directory
        self.previous_files_dir = "LastSaved/" # contains the recognition from the previous recognition
        self.faceDB = "faceDB"
        """END OF FILES"""
        
        self.node_names = ["I", "F", "G", "A", "H", "T"] # 'I' for identity, 'F' for face, 'G' for gender, 'H' for height, 'T' for time of interaction. Don't change the structure, i.e. if a new parameter is to be added, add it to the end of the list
        self.g_labels = ["Female", "Male"]
        self.unknown_var = "0" # id of the unknown variable
        
        self.prob_threshold = 1.0e-75 # the probability below prob_threshold is regarded as prob_threshold. This threshold removes the overall probability to go to 0 in the network.
        self.max_threshold = 0.99 # confidence above max_threshold is set to max_threshold, to maintain Gaussian curves, and to prevent probabilities from going to zero
        self.gender_recognition_rate = self.max_threshold
        self.conf_threshold = 1 - self.max_threshold # the recognition confidence below 0.01 is regarded as 0.01 and uniform distribution is used
        self.face_recognition_rate = 0.9 # for initialisation of likelihoods: P(F=i|I=i) = face_recognition_rate, P(F=j|I=i) = (1-face_recognition_rate)/(num_people-1)

        """THE CONSTANTS OF THE SYSTEM: CAN BE CHANGED DEPENDING ON THE CONTEXT OR THE ALGORITHM USED (NOT NECESSARY)"""
        self.num_recog_min = 5 # for the first num_recog_min recognitions, the identity is estimated as unknown (independent of the number of people in the dataset)

        self.age_min = 0 # min age that can be detected (NAOqi) 
        self.age_max = 75 # max age that can be detected (NAOqi)
        self.age_bin_size = 1 # bin_size = 1 means each age is counted as one bin
#         self.stddev_age = 9.3 # stddev found from data (getAgeStddev function)
        z_age = self.normppf(self.max_threshold + (1-self.max_threshold)/2.0)
        self.stddev_age = 0.5/z_age # standard deviation of age

        self.height_min = 50 # min height that can be detected (NAOqi)
        self.height_max = 240 # max height that can be detected (NAOqi)
        self.height_bin_size = 1 # bin_size = 1 means each height is counted as one bin
        self.stddev_height = 6.3 # stddev found from data (getHeightStddev function)
                
        self.period = 30 # time is checked every 30 minutes 
        self.stddev_time = 60/self.period # 60 minutes
        self.time_min = 0
        self.time_max = (7*24*60/self.period) -1 # 7(days)*24(hours)*60(minutes)/self.period ( = num_time_slots)

# #        self.l_labels = ["Kitchen", "Office"] 
        """END OF CONSTANTS"""
        
        """OPTIMISED PARAMETERS"""
        self.face_recog_threshold = 0.4 # threshold of the face recognition (optimised for NAOqi ALFaceDetection recognition)
        self.quality_threshold = 0.037 # threshold for the quality of the identity estimate (quality = highest_prob - second_highest_prob * num_people) (optimised, see weights for description)
        self.weights = [1.0, 0.044, 0.538, 0.136, 0.906] # [face_weight, gender_weight, age_weight, height_weight, time_weight] (optimised weights for hybrid normalisation with no online learning on IMDB dataset with Nall_uniformT -uniform time)    
        self.update_prob_method = "none" # method for online learning: none, evidence (MMIBN-OL), sum, avg
        self.update_prob_unknown_method = "none" # method for online learning for unknown state: none, evidence (MMIBN-OL), sum, avg
        self.isUpdateFaceLikelihoodsEqually = False # if is equal likelihoods when update method is none
        self.update_partial_params = None # online learning for only specified parameters! self.update_partial_params = None if all parameters are to be learned, e.g. ["A", "T"]
        

        self.norm_method = "hybrid" # normalisation method: norm-sum, minmax, softmax, tanh, hybrid
        if self.norm_method == "hybrid":
            self.evidence_norm_methods = ["norm-sum", "norm-sum", "tanh", "norm-sum", "softmax"]
        else:
            self.evidence_norm_methods = [None for _ in self.node_names[1:]] # the normalisation method will be set to the chosen normalisation method for all parameters for setting evidence
        self.isEvidenceMethodsSet = False
        """END OF OPTIMISED PARAMETERS"""

        """OTHER PARAMETERS (SET)"""
        self.apply_weight_method = "pow" # method for applying weight: (pow works best) pow, invpow, mult
        self.apply_accuracy_method = "none" # method for applying accuracy: (none works best) none, pow, invpow, mult. Currently "mult" doesn't effect the results, because face recognition weight is 1.0 and results are normalised)
        self.update_I_method = "equal" # method for updating prior of Identity node in online learning: "equal" (P(I=i) is equal for all i), "sequential" ('sequential updating'), "occurrences" (P(I=i) is higher when the number of occurrences of i is higher)
        """END OF OTHER PARAMETERS"""

        """PARAMETERS FOR MULTIPLE IMAGES FOR EACH RECOGNITION""" 
        self.isMultipleRecognitions = False # False for single image recognition, True for multiple image recognition
        self.def_num_mult_recognitions = 3 # defined number of recognitions (images) per each recognition
        self.num_mult_recognitions = self.def_num_mult_recognitions
        self.mult_recognitions_list = []
        self.ie_list = []
        self.evidence_list = []
        self.recog_results_list = []
        self.analysis_data_list = []
        """END OF PARAMETERS FOR MULTIPLE IMAGES FOR EACH RECOGNITION"""
        
        self.isDBinCSV = True # is DB in CSV format or in MongoDB format
        self.isSaveRecogFiles = True # False only for optimisation, otherwise, files should be saved for analysis
        self.qualityCoefficient = None # if not None, then the quality formula becomes: quality = (two_largest[0] - two_largest[1]) * self.qualityCoefficient, which replaces the self.num_people
        
        self.isDebugMode = False # print values and errors if True
        self.isLogMode = False # save Analysis.json if True
            
        """ROBOT PARAMETERS"""
        self.useSpanish = False # speak Spanish (for Colombia experiments)
        self.isSpeak = True # if False, the robot will not speak
        self.isMemoryRobot = True # is memory used or not (if False, calculations in confirmPersonIdentity will be skipped, and the identity will not be demanded)
        self.isMemoryOnRobot = False # is the memory on the robot (for experiments with the robot using recognitionModule, this should be True - except in Colombia)
        self.isImageFromTablet = False  # is image received from the tablet or taken by the robot
        """END OF ROBOT PARAMETERS"""
        
        """INITIALISATIONS"""
        self.r_bn = None # Bayesian network
        self.ie = None # Bayesian network inference
        self.identity_est = "" # estimated identity
        self.recog_results = [] # recognition evidence
        self.nonweighted_evidence = [] # nonweighted evidence = recog_results
        self.quality_estimate = -1 # quality of estimation
        self.isRegistered = True # is user registered (enrolled)
        self.isBNLoaded = False # is Bayesian network loaded (to avoid reloading the BN during execution of the code for time purposes)
        self.isUnknownCondition = False # is the user estimated as unknown because Q < Q_threshold
        self.imageToCopy = None # image to copy
        self.num_recognitions = 0 # number of recognitions
        self.isBNSaved = False # is Bayesian network saved (to avoid saving the BN during execution of the code for time purposes)
        self.i_labels = []
        """END OF INITIALISATIONS"""
        

    #---------------------------------------------FUNCTIONS FOR SETTING PARAMETERS OF THE SYSTEM---------------------------------------------# 
        
    def setWeights(self, face_weight, gender_weight, age_weight, height_weight, time_weight):
        """Set weights of the network (default: [1.0, 0.044, 0.538, 0.136, 0.906] for no online learning and hybrid normalisation)"""
        self.weights = [face_weight, gender_weight, age_weight, height_weight, time_weight]
    
    def setParamWeight(self, weight, param):
        """Set weight of specified parameter (weight in the range [0.0, 1.0], param: "F", "G", "A", "H", "T")"""
        if param == "F":
            self.weights[0] = weight
        elif param == "G":
            self.weights[1] = weight
        elif param == "A":
            self.weights[2] = weight
        elif param == "H":
            self.weights[3] = weight
        elif param == "T":
            self.weights[4] = weight
#         elif param == "L":
#             self.weights[5] = weight
        
    def setFaceRecognitionThreshold(self, face_threshold):
        """Set face recognition threshold (0.4 default). Threshold of the face recognition (optimised for NAOqi ALFaceDetection recognition)"""
        self.face_recog_threshold = face_threshold

    def setProbThreshold(self, threshold):
        """Set probability threshold (1.0e-75 default). 
        The probability below prob_threshold is regarded as prob_threshold. This threshold removes the overall probability to go to 0 in the network."""
        self.prob_threshold = threshold
        
    def setQualityThreshold(self, qualityThreshold):
        """Set quality threshold (0.037 default - optimised)"""
        self.quality_threshold = qualityThreshold

    def setMaximumThreshold(self, max_threshold):
        """Set maximum threshold (0.99 default). Confidence above max_threshold is set to max_threshold, 
        to maintain Gaussian curves, and to prevent probabilities from going to zero"""
        self.max_threshold = max_threshold
        
    def setFaceRecognitionRate(self, face_rate):
        """Set face recognition rate (0.9 default). 
        For initialisation of the likelihood (P(F=i|I=i) = self.face_recognition_rate, P(F=j|I=i) = (1-self.face_recognition_rate)/(num_people-1))"""
        self.face_recognition_rate = face_rate
        
    def setGenderRecognitionRate(self, gender_rate):
        """Set gender recognition rate (= max_threshold in default)"""
        self.gender_recognition_rate = gender_rate
    
    def setApplyWeightMethod(self, method):
        """Set apply weight method: pow (default), invpow, mult)"""
        self.apply_weight_method = method
    
    def setApplyAccuracyMethod(self, method):
        """Set method to apply accuracy of face recognition: none (default), pow, invpow, mult"""
        self.apply_accuracy_method = method
        
    def setNormMethod(self, method):
        """Set normalisation method: hybrid (default), norm-sum, minmax, softmax, tanh"""
        self.norm_method = method
        if self.norm_method == "hybrid":
            self.evidence_norm_methods = ["norm-sum", "norm-sum", "tanh", "norm-sum", "softmax"]
        elif not self.isEvidenceMethodsSet:
            self.evidence_norm_methods = [None for _ in self.node_names[1:]]
    
    def setEvidenceNormMethods(self, methods):
        """Set normalisation method: hybrid (default), norm-sum, minmax, softmax, tanh"""
        for method_c in range(0, len(methods)):
            self.evidence_norm_methods[method_c] = methods[method_c]
        self.isEvidenceMethodsSet = True
        
    def setUpdateMethod(self, method, unknown_update_method = None):
        """Set update (online learning) method: none (default), evidence, sum, avg. 
        Update method for I = unknown can be specified by unknown_update_method as well (none: default, evidence, sum, avg). 
        If not set for unknown, it would default to the online learning method of the system."""
        self.update_prob_method = method
        if unknown_update_method is not None:
            self.update_prob_unknown_method = unknown_update_method
        else:
            self.update_prob_unknown_method = method
            
    def setUpdatePartialParams(self, params_list = None):
        """Update (online learning) of the specified parameters only: None (all params will be learned), or a list (e.g. ["T"] - default)"""
        self.update_partial_params = params_list
    
    def setUpdateFaceLikelihoodsEqually(self, isUpdateFaceLikelihoodsEqually = True):
        """If is isUpdateFaceLikelihoodsEqually = True and update_prob_method is none, then after a new person is enrolled, face likelihoods will be updated as
        P(F=f|I=i) = face_recognition_rate^weight_F if f=i, P(F=f|I=i) = ((1 - face_recognition_rate)/(num_people-1))^weight_F  if f!=i
        
        Otherwise (isUpdateFaceLikelihoodsEqually = False), then ONLY unknown likelihood will be updated as such.
        If update_prob_method is not none, then likelihoods will be updated according to the update method."""
        self.isUpdateFaceLikelihoodsEqually = isUpdateFaceLikelihoodsEqually
        
    def setDefinedNumMultRecognitions(self, num_mult_recognitions):
        """Set defined number of recognitions for multiple image recognition (3 default)"""
        self.def_num_mult_recognitions = num_mult_recognitions

    def setNumberImagesPerRecognition(self, isMultRecognitions, num_mult_recognitions = None):
        """Set number of images per recognition. If isMultRecognitions False (default), single image recognition will be made, if True, num_mult_recognitions can be specified (3 default)"""
        self.isMultipleRecognitions = isMultRecognitions
        if isMultRecognitions:
            self.def_num_mult_recognitions = num_mult_recognitions
            self.num_mult_recognitions = self.def_num_mult_recognitions
    
    def setQualityCoefficient(self, qualityCoefficient):
        """Set quality coefficient for the equation: 
        Quality = highest_prob - second_highest_prob * qualityCoefficient. Default num_people is used for the coefficient"""
        self.qualityCoefficient = qualityCoefficient
        
    def setSaveRecogFiles(self, isSaveRecogFiles):
        """Set isSaveRecogFiles (default: True). If True, the files are saved, make False only for optimisation."""
        self.isSaveRecogFiles = isSaveRecogFiles
        
    def setDebugMode(self, mode = True):
        """Set debug mode. If True, print values and errors"""
        self.isDebugMode = mode
        
    def setLogMode(self, mode = True):
        """Set log mode. If True, saves analysis file"""
        self.isLogMode = mode
        
    def setSilentMode(self):
        """When called, the robot does not speak"""
        self.isSpeak = False

    def setImageToCopy(self, imagePath):
        """Set the path of the image to copy"""
        self.imageToCopy = imagePath

    #---------------------------------------------LOAD/SAVE BN/DB ---------------------------------------------# 
    
    def loadBN(self, recog_file, recogniser_csv_file, initial_recognition_file):
        """Load Bayesian Network and database or create a new BN  if it doesn't exist (from scratch or learn from data files if they exist)"""
        
        self.recog_file = recog_file
        self.recogniser_csv_file = recogniser_csv_file
        start_load_bn = time.time()

        self.loadDB(self.db_file)
        
        self.num_recognitions = sum([self.occurrences[i][0] for i in range(1, len(self.occurrences))])       
        self.isBNLoaded = True
        
        if os.path.isfile(recog_file):
            self.r_bn = gum.loadBN(recog_file)
            self.loadVariables()
            end_load_bn = time.time()
            
            if self.isDebugMode:
                print "time to load network:" + str(end_load_bn - start_load_bn)
                
        elif self.num_people > 1:
            if self.num_recognitions == 0:
                self.r_bn=gum.BayesNet('RecogniserBN')
                self.addNodes()
                self.addArcs()
                self.addCpts() 
            else:
                db_list = self.clearDB()
                self.learnFromFile(db_list=None, db_file = self.db_file, init_recog_file = initial_recognition_file, final_recog_file = recogniser_csv_file)
                
    def saveBN(self, recog_file = None, num_recog = None):
        """Save the Bayesian network to the file"""

        if self.r_bn is not None:
            if recog_file is None:
                recog_file = self.recog_file
            else:
                self.recog_file = recog_file
            start_save_bn = time.time()

            if self.isMultipleRecognitions:
                if num_recog == self.num_mult_recognitions - 1:
                    gum.saveBN(self.r_bn, recog_file)
                    if self.isDebugMode:
                        print "time for agrum save:" + str(time.time() - start_save_bn)
            else:
                gum.saveBN(self.r_bn, recog_file)
                if self.isDebugMode:
                    print "time for agrum save:" + str(time.time() - start_save_bn)
            
            self.isBNSaved = True

    def loadVariables(self):
        """Load variables of the network to self.I, self.F, etc. Get their IDs into node_ids. Load the likelihoods into cpt_matrix (for faster execution)"""
        self.I = self.r_bn.idFromName("I")
        self.F = self.r_bn.idFromName("F")
        self.G = self.r_bn.idFromName("G")
        self.A = self.r_bn.idFromName("A")
        self.H = self.r_bn.idFromName("H")
        self.T = self.r_bn.idFromName("T")
# #         self.L = self.r_bn.idFromName("L")
        
        self.node_ids = {"I": self.I, "F": self.F, "G": self.G, "A": self.A, "H": self.H, "T": self.T}
        self.cpt_matrix = []
        for counter in range(0, len(self.i_labels)):
            person_cpts = []
            for name_param in self.node_names[1:]:
                id_v = self.node_ids[name_param]
                person_cpts.append(self.r_bn.cpt(id_v)[{'I':counter}])
            self.cpt_matrix.append(person_cpts)

    def clearDB(self):
        """Clear dabatase and return the old database if exists"""
        old_db = None
        if self.i_labels:
            old_db = [self.i_labels, self.names, self.genders, self.ages, self.heights, self.times, self.occurrences]
        self.i_labels = []
        self.names = []
        self.genders = []
        self.ages = []
        self.heights = []
        self.times =[]       
        self.occurrences = []
        self.addUnknown()
        self.num_people = len(self.i_labels)     
        return old_db
               
    def loadDB(self, db_file = None):
        """Reset and load db from file or mongo db"""
        self.i_labels = []
        self.names = []
        self.genders = []
        self.ages = []
        self.heights = []
        self.times =[]       
        self.occurrences = []
        self.num_people = 0
        if not self.isSaveRecogFiles:
            self.addUnknown()
            self.num_people = len(self.i_labels)
        elif self.isDBinCSV:
            self.loadDBFromCSV(db_file)
        else:
            bla = ""
#             db_handler = db.DbHandler()
#             self.loadDBFromMongo(db_handler)
#         self.printDB()

    def loadDBFromCSV(self, db_file):
        """Load db from csv file"""
        if os.path.isfile(db_file):
            self.db_df = pandas.read_csv(db_file, dtype={"id": object, "name": object}, converters={"times": ast.literal_eval, "occurrence": ast.literal_eval})
            self.i_labels = self.db_df['id'].values.tolist()
            self.names = self.db_df['name'].values.tolist()
            self.genders = self.db_df['gender'].values.tolist()
            if 'age' in self.db_df.columns: # for preventing compatibility issues
                self.ages = self.db_df['age'].values.tolist()
            elif 'birthYear' in self.db_df.columns:
                self.ages = []
                for birth_year in self.db_df['birthYear'].values.tolist():
                    self.ages.append(self.getAgeFromBirthYear(birth_year))
            self.heights = self.db_df['height'].values.tolist()
            self.times = [] 
            ti = self.db_df['times'].values.tolist()
            for t in ti:
                times_users = []
                for tt in t:
                    times_users.append(tt[:2])
                self.times.append(times_users)
            self.occurrences = self.db_df['occurrence'].values.tolist() #[num_occurrence, num_images_for_registering, num_total_images]
        
        self.addUnknown()
        
        self.num_people = len(self.i_labels)
#         self.printDB()

    def loadDBFromMongo(self, db_handler):
        """Load db from mongo db"""
        p = db_handler.get_all_patients()
        counter_p = 0
        for a in p:
            self.i_labels.append(str(a["Id_number"]))
            self.names.append(str(a["name"]))
            self.genders.append(str(a["gender"]))
            if "age" in a.keys():
                self.ages.append(int(a["age"]))
            elif "birthYear" in a.keys():
                self.ages.append(self.getAgeFromBirthYear(int(a["birthYear"])))
            self.heights.append(float(a["height"]))
            times_users = []
            for tt in a["times"]:
                times_users.append([tt.time().strftime('%H:%M:%S'), tt.isoweekday()])
            self.times.append(times_users)
            counter_p = counter_p + 1
            self.occurrences.append(a["occurrence"])
        self.addUnknown()
        self.num_people = len(self.i_labels)

    def updateDB(self, db_file, p_id):
        """Update the occurrence of user in db"""
        if self.isDBinCSV:
            self.db_df.loc[self.db_df['id'] == p_id, 'occurrence'] = str(self.occurrences[self.i_labels.index(p_id)])
            self.db_df.to_csv(db_file, index=False)
        else:
            # TODO: fill it here if not using CSV (we suggest CSV for ease of data manipulation and collection)
            pass

    def loadDummyData(self):
        """Load dummy data to test the system"""
        self.i_labels = ["1","2","3"]
        self.names = ["Jane","James","John"]
        self.genders = ["Female","Male","Male"]
        self.ages = [25,25,25]
        self.heights = [168, 168, 168]
        self.times =[[["11:00:00",1]], [["11:00:00",1]], [["11:00:00",1], ["12:00:00",3]]]
        self.occurrences = [[1,1,1], [2,3,6],[1,2,2]]    
        self.num_people = 3
        
    #---------------------------------------------INITIALISE NODES/LIKELIHOODS ---------------------------------------------# 

    def addNodes(self):  
        """Add nodes to the network"""
        # Identity node
        self.identity_node = gum.LabelizedVariable("I","Identity",0)
        for counter in range(0, len(self.i_labels)):
            self.identity_node.addLabel(self.i_labels[counter])       
        self.I = self.r_bn.add(self.identity_node)
        
        # Face node
        self.face_node = gum.LabelizedVariable("F","Face",0)
        for counter in range(0, len(self.i_labels)):
            self.face_node.addLabel(self.i_labels[counter]) 
        self.F = self.r_bn.add(self.face_node)

        # Gender node
        self.gender_node = gum.LabelizedVariable("G","Gender",0)
        for counter in range(0, len(self.g_labels)):
            self.gender_node.addLabel(self.g_labels[counter])
        self.G = self.r_bn.add(self.gender_node)
        
        # Age node
        self.age_node = gum.RangeVariable("A","Age",self.age_min,self.age_max)
        self.A = self.r_bn.add(self.age_node)      
        
        # Height node
        self.height_node = gum.RangeVariable("H","Height",self.height_min,self.height_max)
        self.H = self.r_bn.add(self.height_node)
        
        # Time node
        self.time_node= gum.RangeVariable("T","Time",self.time_min,self.time_max)
        self.T = self.r_bn.add(self.time_node)
        
        self.node_ids = {"I": self.I, "F": self.F, "G": self.G, "A": self.A, "H": self.H, "T": self.T}
#         gnb.showBN(self.r_bn)
        
# # #         # Location node
# # #         self.location = gum.LabelizedVariable("L","Location",0)
# # #         for counter in range(0, len(self.l_labels)):
# # #             self.location.addLabel(self.l_labels[counter])
# # #         self.L = self.r_bn.add(self.location)
        
        
    def addArcs(self):
        """Add arcs between nodes"""
        self.r_bn.addArc(self.I,self.F)
        self.r_bn.addArc(self.I,self.G)
        self.r_bn.addArc(self.I,self.A)
        self.r_bn.addArc(self.I,self.H)
        self.r_bn.addArc(self.I,self.T)
# #  #       self.r_bn.addArc(self.T,self.L)
# #  #       self.r_bn.addArc(self.I,self.L)

        
    def addCpts(self):
        """Initialise likelihoods for unknown state and the remaining users"""
        self.cpt_matrix = []
        # P(I)
        self.r_bn.cpt(self.I)[:] = self.updatePriorI()
        index_unknown = self.i_labels.index(self.unknown_var)
        # P(F|I), P(G|I), P(A|I), P(H|I), P(T|I)
        for counter in range(0, len(self.i_labels)):
            if counter == index_unknown:
                self.addUnknownLikelihood(self.r_bn)
            else:
                self.addLikelihoods(counter)
          
    def addLikelihoods(self, p_index):
        """Initialise likelihoods for users (NOT for unknown): 
        P(F=f|I=i) = face_recognition_rate^weight_F if f=i
        P(F=f|I=i) = ((1 - face_recognition_rate)/(num_people-1))^weight_F  if f!=i
        
        P(G="Female"|I=i) =  gender_recognition_rate^weight_G if user i is "Female"
        P(G="Male"|I=i) =  (1-gender_recognition_rate)^weight_G
        
        Age (P(A = a|I = i)), height (P(H = h|I = i)) and time (P(T = t|I = i)) are estimated from 
        discretised and normalised normal distributions using getCurve function, by using the ground truth values of the person, 
        (real age, real height, time of the interaction at registration (or multiple times if they are saved in the database)).
        
        The normalisation method is "norm-sum" for all parameters for initialisation.
        """
        
        person_cpt_list = []
        # P(F|I)  

#         li_f = [ self.applyWeight(self.init_min_threshold,self.weights[0]) for x in range(0, len(self.i_labels))]
#         li_f[p_index] = self.applyWeight(1.0, self.weights[0])

#         li_f[p_index] = self.applyWeight(1 - ((len(self.i_labels)-1)*self.applyWeight(self.init_min_threshold,self.weights[0])),self.weights[0])
        
        li_f = [self.applyWeight((1 - self.face_recognition_rate)/(len(self.i_labels)-1),self.weights[0]) for x in range(0, len(self.i_labels))]
        li_f[p_index] = self.applyWeight(self.face_recognition_rate, self.weights[0])
        
        li_f = self.normaliseSum(li_f)
        # what it does: self.r_bn.cpt(self.I)[{'F':0}]=1 SAME THING AS: self.r_bn.cpt(self.I)[{'F':self.unknown_var}]=[0.5,0.5]
        self.r_bn.cpt(self.F)[{'I':self.i_labels[p_index]}] = li_f[:]
        person_cpt_list.append(li_f[:])
        
        # P(G|I)  
#         li_g = [self.applyWeight(self.init_min_threshold, self.weights[1]), self.applyWeight(self.init_min_threshold, self.weights[1])]
#         if self.genders[p_index] == self.g_labels[0]:
#             li_g[0] = self.applyWeight(1-self.init_min_threshold, self.weights[1])
#         else:
#             li_g[1] = self.applyWeight(1-self.init_min_threshold, self.weights[1])

        li_g = [self.applyWeight(1 - self.gender_recognition_rate, self.weights[1]), self.applyWeight(1 - self.gender_recognition_rate, self.weights[1])]
        if self.genders[p_index] == self.g_labels[0]:
            li_g[0] = self.applyWeight(self.gender_recognition_rate, self.weights[1])
        else:
            li_g[1] = self.applyWeight(self.gender_recognition_rate, self.weights[1])
        li_g = self.normaliseSum(li_g)
        self.r_bn.cpt(self.G)[{'I':self.i_labels[p_index]}] = li_g[:]
        person_cpt_list.append(li_g[:])
        
        # P(A|I)      
        age_curve_pdf = self.getCurve(mean = self.ages[p_index], stddev = self.stddev_age, min_value = self.age_min, max_value = self.age_max, weight = self.weights[2], norm_method = "norm-sum")
        self.r_bn.cpt(self.A)[{'I':self.i_labels[p_index]}] = age_curve_pdf[:]
        person_cpt_list.append(age_curve_pdf[:])
        
        # P(H|I)        
        height_curve_pdf = self.getCurve(mean = self.heights[p_index], stddev = self.stddev_height, min_value = self.height_min, max_value = self.height_max, weight = self.weights[3], norm_method = "norm-sum")
        self.r_bn.cpt(self.H)[{'I':self.i_labels[p_index]}] = height_curve_pdf[:]
        person_cpt_list.append(height_curve_pdf[:])
        
        # P(T|I)
        time_curve_total_pdf = []
        for t_count in range(0, len(self.times[p_index])):
            time_curve_pdf = self.getCurve(mean = self.getTimeSlot(self.times[p_index][t_count]), stddev = self.stddev_time, min_value = self.time_min, max_value = self.time_max, weight = self.weights[4], norm_method = "norm-sum")
            if t_count == 0:
                time_curve_total_pdf = time_curve_pdf[:]
            else:
                time_curve_total_pdf = [x + y for x, y in zip(time_curve_total_pdf, time_curve_pdf)]
                
        time_curve_total_pdf = self.normaliseSum(time_curve_total_pdf)
        self.r_bn.cpt(self.T)[{'I':self.i_labels[p_index]}] = time_curve_total_pdf[:]
        person_cpt_list.append(time_curve_total_pdf[:])
        self.cpt_matrix.append(person_cpt_list)

    def addUnknown(self):
        """Add values for unknown state in the database: (the values do not represent anything)
        ID is '0', name is 'unknown', gender is 'not-known', age is '35', height is '165', time is '00:00:00' on Monday,
        occurrence [1,0,1] 
        """
        
        self.i_labels.insert(0, self.unknown_var)
        self.names.insert(0, "unknown")
        self.genders.insert(0, "not-known")
        self.ages.insert(0, 35)
        self.heights.insert(0, 165)
        self.times.insert(0, [["00:00:00",1]])
        
        count_unknown = 0
        count_unknown_images = 0
        
        # NOTE: start range from 1: first person face recognition data is not added to unknown probability 
        #(face recognition result when there is noone in the db will be [], so these values will not be added to unknown face posterior)
        for i in range(1, len(self.i_labels)-1):
            if self.occurrences[i][0] > 0:
                count_unknown += 1
                count_unknown_images += self.occurrences[i][1]
        self.occurrences.insert(0,[count_unknown, 0, count_unknown_images])
             
    def addUnknownLikelihood(self, bn):
        """Add likelihoods for unknown state. P(F=f|I='0') is set in the way that is used for the other states.
        P(G=g|I='0') = 0.5 (equally likely to be Female or Male)
        P(A=a|I='0'),P(H=h|I='0'), P(T=t|I='0') have uniform distributions.
        """
        
        person_cpt_list = []
        
        counter = self.i_labels.index(self.unknown_var)

        # DONT USE THIS (0.5 FOR UNKNOWN, 0.5/(1-num_people) FOR THE REST)! IT MAKES FAR (FALSE ALARM RATE) WORSE
        # gives higher false positive! use the other one instead
#         li_f_unnorm = [ self.applyWeight(0.5/(len(self.i_labels)-1),self.weights[0]) for x in range(0, len(self.i_labels))]
#         li_f_unnorm[counter] = self.applyWeight(0.5, self.weights[0])

        # P(F|I) same way as the likelihoods for other states 
        li_f_unnorm = [self.applyWeight((1 - self.face_recognition_rate)/(len(self.i_labels)-1),self.weights[0]) for x in range(0, len(self.i_labels))]
        li_f_unnorm[counter] = self.applyWeight(self.face_recognition_rate, self.weights[0])
        li_f = self.normaliseSum(li_f_unnorm)
        bn.cpt(self.F)[{'I':self.unknown_var}] = li_f[:]
        person_cpt_list.append(li_f[:])
        
        # P(G|I) : Equally likely to be male or female
        li_g = [0.5, 0.5]
        bn.cpt(self.G)[{'I':self.unknown_var}] = li_g[:]
        person_cpt_list.append(li_g[:])
            
        # P(A|I) : Uniform distribution for unknown age
        li_a = self.uniformDistribution(self.age_min, self.age_max)
        bn.cpt(self.A)[{'I':self.unknown_var}] = li_a[:]
        person_cpt_list.append(li_a[:])
        
        # P(H|I) : Uniform distribution for unknown height
        li_h = self.uniformDistribution(self.height_min, self.height_max)
        bn.cpt(self.H)[{'I':self.unknown_var}] = li_h[:]
        person_cpt_list.append(li_h[:])
        
        # P(T|I) : Uniform distribution for any time  
        li_t = self.uniformDistribution(self.time_min, self.time_max)
        bn.cpt(self.T)[{'I':self.unknown_var}] = li_t[:]
        person_cpt_list.append(li_t[:])
        
        self.cpt_matrix.append(person_cpt_list)

    #---------------------------------------------UPDATE NODES/LIKELIHOODS ---------------------------------------------# 

    def addPersonToBN(self, person, isSaveDB = True):
        """
        Add new user to BN and to the database
        Get from input (for adding people into db) (person = ["1", Jane", "Female", 26, 175, [arrayOfTimesOfSessionsInDateTimeFormat], [1,2,2]])
        If no BN exists, create the Bayesian network
        """
        if not self.isBNLoaded:            
            self.loadBN(self.recog_file, self.recogniser_csv_file, self.initial_recognition_file)

        if person[0] in self.i_labels:
            logging.debug("The user is already in the database.")
        else:
            start_time_update_data = time.time()
            self.updateData(person)
            end_time_update_data = time.time()

            if self.isDebugMode:
                print "time to update data:" + str(end_time_update_data - start_time_update_data)
            if isSaveDB and self.isDBinCSV and self.isSaveRecogFiles:
                self.saveDB(self.db_file, person)
                if self.isDebugMode:
                    print "time to save db:" + str(time.time() - end_time_update_data)
                
        if self.num_people == 2:
            self.r_bn=gum.BayesNet('RecogniserBN')
            self.addNodes()
            self.addArcs()
            self.addCpts()
        elif self.num_people > 2:
            start_add_person_to_nodes = time.time()
            if self.r_bn.variableFromName("I").toLabelizedVar().isLabel(person[0]):
                logging.debug("The user is already in the network.")
            else:
                mid_add_person_to_nodes = time.time()
                self.updateNodes(person[0])
                if self.isDebugMode:
                    print "time to update nodes:" + str(time.time() - mid_add_person_to_nodes)
   
    def updateData(self, person):
        """Update the database with the new user information:
        add the ID to 'i_labels', add name to 'names', add gender to 'genders', 
        add age to 'ages', add height to 'heights', add time of interaction(s) to 'times', and add occurrence to occurrences.
        Increment num_people
        """
        
        self.i_labels.append(str(person[0]))
        self.names.append(str(person[1]))
        self.genders.append(person[2])
        if person[3] > 1900: # if is birthyear
            self.ages.append(self.getAgeFromBirthYear(person[3]))
        else:
            self.ages.append(person[3])
        self.heights.append(person[4])
        times_users = []
        if self.isDBinCSV:
            for tt in person[5]:
                times_users.append(tt[:2])
        else:
            for tt in person[5]:
                times_users.append([tt.time().strftime('%H:%M:%S'), tt.isoweekday()])
        self.times.append(times_users)
        if self.isMultipleRecognitions:
            self.occurrences.append([0, self.num_mult_recognitions, 0])
        else:
            self.occurrences.append([0, 1, 0])

        self.num_people += 1

    def updateNodes(self, p_id):
        """Call the function when a new person added is to the db
        CPT is a property of the BN and not the variable, therefore, 
        to add a new state to a node, it is necessary to copy the previous CPT,
        change it accordingly, 
        erase the nodes F and I, add a new state to F and I to recreate them, 
        add the changed node back to the BN, add the arcs, and add the changed CPT to it
        change the CPT of the child nodes
        """
        
        prev_face_recog_rate = self.face_recognition_rate
        
#         init_I_priors = self.r_bn.cpt(self.I)[:]

        # Erase I and F
        self.r_bn.erase(self.I)
        self.r_bn.erase(self.F)        
        
        # Change and add nodes
        # Face node
        self.face_node = gum.LabelizedVariable("F","Face",0)

        for counter in range(0, len(self.i_labels)):
            self.face_node.addLabel(self.i_labels[counter]) 
        self.F = self.r_bn.add(self.face_node)
        self.node_ids["F"] = self.F
        # Identity node
        self.identity_node = gum.LabelizedVariable("I","Identity",0)
        for counter in range(0, len(self.i_labels)):
            self.identity_node.addLabel(self.i_labels[counter])       
        self.I = self.r_bn.add(self.identity_node)        
        self.node_ids["I"] = self.I
        
        self.addArcs()
        
        # Change CPT
        updated_cpt_I = []
        
        # copy previous likelihoods back into the network for G, A, H, and T
        self.r_bn.cpt(self.G)[:-1] = [i[1] for i in self.cpt_matrix]
        self.r_bn.cpt(self.A)[:-1] = [i[2] for i in self.cpt_matrix]
        self.r_bn.cpt(self.H)[:-1] = [i[3] for i in self.cpt_matrix]
        self.r_bn.cpt(self.T)[:-1] = [i[4] for i in self.cpt_matrix]

        for counter in range(0, len(self.i_labels)):
            if counter < len(self.i_labels) - 1:
                if self.isUpdateFaceLikelihoodsEqually and ((self.update_prob_unknown_method == "none" and counter == self.i_labels.index(self.unknown_var)) or self.update_prob_method == "none" or (self.update_partial_params is not None and "F" not in self.update_partial_params)):
                    # THIS UPDATES ALL LIKELIHOODS TO BE (IF NO ONLINE LEARNING):  
                    # P(F=f|I=i) = face_recognition_rate^weight_F if f=i, P(F=f|I=i) = ((1 - face_recognition_rate)/(num_people-1))^weight_F  if f!=i
                    # BUT IT DOESN'T PERFORM AS GOOD AS UPDATING AS IN 'ELSE' CONDITION
                    self.cpt_matrix[counter][0] = self.getEqualFaceLikelihoods(counter)
                elif not self.isUpdateFaceLikelihoodsEqually and ((self.update_prob_unknown_method == "none" or self.update_prob_method == "none") and counter == self.i_labels.index(self.unknown_var)):
                    # UPDATES ONLY UNKNOWN LIKELIHOOD TO BE (IF NO ONLINE LEARNING):  
                    # P(F=f|I=i) = face_recognition_rate^weight_F if f=i, P(F=f|I=i) = ((1 - face_recognition_rate)/(num_people-1))^weight_F  if f!=i
                    self.cpt_matrix[counter][0] = self.getEqualFaceLikelihoods(counter)
                else:
                    # If the user is never seen before, update the likelihood of that user to =(1-face_recognition_rate)/(num-people-1)
                    if self.occurrences[counter][0] == 0:
                        for ff in range(0, len(self.i_labels)-1):
                            if np.isclose(self.cpt_matrix[counter][0][ff], (1-prev_face_recog_rate)/(len(self.i_labels)-2)):
                                self.cpt_matrix[counter][0][ff] = (1-self.face_recognition_rate)/(len(self.i_labels)-1)
                        updated_cpt_F = self.cpt_matrix[counter][0][:]
                    else:
                        # if the user is previously seen, then update the likelihoods by computing the original likelihood by multiplying with occurrence 
                        # and then adding the new user likelihood, then normalising. 
                        # i.e. P(F=f|I=i)_total = [P(F=f|I=i)*num_occurrence(f)].append(1-face_recognition_rate)/(num_people-1)) and normalise 
                        
                        if self.update_prob_method == "avg":
                            occur = self.occurrences[counter][0] + 1
                        else: #self.update_prob_method == "sum" or self.update_prob_method == "evidence" or self.update_prob_method == "none":
                            occur = self.occurrences[counter][2] + 1
    
                        updated_cpt_F = [i*occur for i in self.cpt_matrix[counter][0] ]
                    updated_cpt_F = np.append(updated_cpt_F, [(1-self.face_recognition_rate)/(len(self.i_labels)-1)]) 
                    updated_cpt_F = self.normaliseSum(updated_cpt_F)
                    self.cpt_matrix[counter][0] = updated_cpt_F[:]
            else:
                self.addLikelihoods(counter)
        self.r_bn.cpt(self.F)[:] = [i[0] for i in self.cpt_matrix]
        
        # update P(I)
        self.r_bn.cpt(self.I)[:] = self.updatePriorI()
#         self.r_bn.cpt(self.I)[:] = self.updatePriorI(p_id, init_I_priors)

                    
    def updatePriorI(self, p_id = None, ie = None):
        """
        update_I_method is 'equal' by default, otherwise the network is biased towards the people seen.
        Method for updating prior of Identity node in online learning: 
        "equal" (P(I=i) is equal for all i), 
        "sequential" ('sequential updating'), 
        "occurrences" (P(I=i) is higher when the number of occurrences of i is higher)
        """
        if self.update_I_method == "equal":
            updated_priors = self.priorIEqualProbabilities()
        elif self.update_I_method == "sequential":
            updated_priors = self.priorISequentialUpdating(ie)
        elif self.update_I_method == "occurrences":
            updated_priors = self.priorIOccurrences(p_id)
        return updated_priors
        
    def priorIOccurrences(self, p_id = None):
        """
        NOT USED!! Reason: when I use this P(I=i) = num_times_i_seen/num_recognitions,
        the network is biased towards the people seen. So it is better to use equal values for P(I=i) = 1/num_people
        """
        prob_values = [occur[0] for occur in self.occurrences]
        if p_id is None:
            if sum(prob_values) == 0:
                prob_values = [1/len(self.i_labels) for i in range(0,len(self.i_labels))]
            else:
                for i_count in range(0, len(self.i_labels)):
                    if self.occurrences[i_count][0] == 0:  
                        prob_values[i_count] = self.init_min_threshold
            prob_values = [i/float(self.num_recognitions) for i in prob_values]
        else:
            index_name = self.i_labels.index(p_id)
            if sum(prob_values) == 0:
                prob_values = [self.init_min_threshold for i in range(0,len(self.i_labels))]
                prob_values[index_name] = 1
            else:            
                for i_count in range(0, len(self.i_labels)):
                    if i_count ==  index_name:
                        prob_values[i_count] += 1
                    elif self.occurrences[i_count][0] == 0:   
                        prob_values[i_count] = self.init_min_threshold

            prob_values = [i/float(self.num_recognitions + 1) for i in prob_values]
        prob_norm = self.normaliseSum(prob_values)
        return prob_norm
        
    def priorISequentialUpdating(self, ie = None):
        """
        NOT USED!! Reason: Biases network towards people met before. 
        According to Sequential Bayesian updating, posterior of the previous calculation can be used as the prior
        """
        if self.isAddPersonToDB:
            prev_cpt = np.array(ie)
            prev_cpt = np.append(prev_cpt, [(1-self.face_recognition_rate)/(len(self.i_labels)-1)])
            return prev_cpt
        return np.array(self.ie.posterior(self.I)[:])

    def priorIEqualProbabilities(self):
        """
        Equal probabilities is given to all, e.g. P(I=i) = 1/num_people
        """
        return [1.0/len(self.i_labels) for i in range(0, len(self.i_labels))]
        
    def getEqualFaceLikelihoods(self, indUser):
        """Returns updated face likelihoods with P(F=f|I=i) = face_recognition_rate^weight_F if f=i, P(F=f|I=i) = ((1 - face_recognition_rate)/(num_people-1))^weight_F  if f!=i"""
        li_f = [self.applyWeight((1 - self.face_recognition_rate)/(len(self.i_labels)-1),self.weights[0]) for x in range(0, len(self.i_labels))]
        li_f[indUser] = self.applyWeight(self.face_recognition_rate, self.weights[0])

        norm_li_f = self.normaliseSum(li_f)
        return norm_li_f
    
    #---------------------------------------------ONLINE LEARNING ---------------------------------------------# 

    def updateProbabilities(self, p_id, ie, evidence, num_recog = None, num_mult_recognitions = None):
        """Online learning of the likelihoods: P(I), P(F|I), P(G|I), P(A|I), P(H|I), P(T|I) in the network."""
        
        if self.isMultipleRecognitions and num_mult_recognitions is None:
            num_mult_recognitions = self.num_mult_recognitions

        if self.update_prob_method == "none":
            # if no online learning, then update the occurrences of the user only
            if not self.isMultipleRecognitions:
                self.occurrences[self.i_labels.index(p_id)][0] += 1
                self.occurrences[self.i_labels.index(p_id)][2] += 1                    
            elif num_recog == num_mult_recognitions -1:
                self.occurrences[self.i_labels.index(p_id)][0] += 1
                self.occurrences[self.i_labels.index(p_id)][2] += num_mult_recognitions
            return
          
        if p_id == self.unknown_var:
            # update ONLY face likelihood for unknown. 
            # The remaining parameters do not change, as the gender, age, height and time should be uniform -> the unknown user can be of any gender, age, height or can come at any time
            iter_list = [self.node_names[1]]
        elif self.update_partial_params is not None:
            # for partial update (only the specified parameters are updated, e.g. only time if update_partial_params = ["T"])
            iter_list = self.update_partial_params
        else:
            # online learning of all parameters (except identity)
            iter_list = self.node_names[1:]

        if self.update_prob_method == "evidence":
            ie.setEvidence({"F":evidence[0], "G":evidence[1], "A":evidence[2], "H":evidence[3], "T":evidence[4], "I":p_id})
            ie.makeInference()

        if self.update_I_method != "equal":
            if not self.isMultipleRecognitions:
                self.r_bn.cpt(self.I)[:] = self.updatePriorI(p_id, ie)
            elif num_recog == num_mult_recognitions -1:
                self.r_bn.cpt(self.I)[:] = self.updatePriorI(p_id, ie)
                
        p_id_index = self.i_labels.index(p_id)
        for name_param in iter_list:
            counter = self.node_names.index(name_param) - 1
            if self.weights[counter] > 0:
                # there is no point in updating if weight is zero, since all values will be equal
                if self.isMultipleRecognitions:
                    prob_values = evidence[counter]
                    if num_recog == 0 and counter == 0:
                        # reset
                        self.prev_update_prob = []
                    if self.update_prob_method == "avg":
                        # get the average of recognition values of the multiple recognitions and then sum with the previous likelihoods, normalise and update
                        if num_recog == 0:
                            self.prev_update_prob.append(prob_values)
                        else:
                            self.prev_update_prob[counter] = list(np.add(self.prev_update_prob[counter], prob_values))
                            
                        if num_recog == num_mult_recognitions - 1:
                            id_v = self.node_ids[name_param]
                            cpt_norm = self.cpt_matrix[p_id_index][counter][:]
                            occur = self.occurrences[self.i_labels.index(p_id)][0] + 1
                            cpt_unnorm = [i*occur for i in cpt_norm]
                            self.prev_update_prob[counter] = self.normaliseSum(self.prev_update_prob[counter])
                            self.prev_update_prob[counter] = list(np.add(self.prev_update_prob[counter], cpt_unnorm))
                            norm_total_prob = self.normaliseSum(self.prev_update_prob[counter])
                            self.r_bn.cpt(id_v)[{'I':p_id}] = norm_total_prob[:] 
                            self.cpt_matrix[p_id_index][counter] = norm_total_prob[:]
                            
                    elif self.update_prob_method == "sum":
                        # sum the previous likelihoods, with the current recognition values and normalise to update
                        if num_recog == 0:
                            id_v = self.node_ids[name_param]
                            prev_prob_norm = self.cpt_matrix[p_id_index][counter][:]
                            occur = self.occurrences[self.i_labels.index(p_id)][2] + 1
                            self.prev_update_prob.append([i*occur for i in prev_prob_norm])
                        
                        self.prev_update_prob[counter] = list(np.add(self.prev_update_prob[counter], prob_values))
                        if num_recog == num_mult_recognitions -1:                        
                            norm_total_prob = self.normaliseSum(self.prev_update_prob[counter])
                            id_v = self.node_ids[name_param]
                            self.r_bn.cpt(id_v)[{'I':p_id}] = norm_total_prob[:]   
                            self.cpt_matrix[p_id_index][counter] = norm_total_prob[:]
                              
                    elif self.update_prob_method == "evidence":
                        # use the inference based on the evidence of all the recognition values and the correct identity to update the likelihoods, by summing with the previous likelihoods and normalising
                        id_v = self.node_ids[name_param]
                        if num_recog == 0:
                            prev_prob_norm = self.cpt_matrix[p_id_index][counter][:]
                            occur = self.occurrences[self.i_labels.index(p_id)][2] + 1
                            self.prev_update_prob.append([i*occur for i in prev_prob_norm])
                        try:
                            prob_values = ie.posterior(id_v)
                            self.prev_update_prob[counter] = list(np.add(self.prev_update_prob[counter], prob_values[:]))

                        except:
                            # if inference can't be made (recognition values are empty - no face detected in the image or problem with recogniser, continue
                            pass
    
                        if num_recog == num_mult_recognitions -1:                        
                            norm_total_prob = self.normaliseSum(self.prev_update_prob[counter])
                            self.r_bn.cpt(id_v)[{'I':p_id}] = norm_total_prob[:]     
                            self.cpt_matrix[p_id_index][counter] = norm_total_prob[:]
    
                else:
                    id_v = self.node_ids[name_param]
                    
                    prev_prob_norm = self.cpt_matrix[p_id_index][counter][:] # faster
                    
                    occur = self.occurrences[self.i_labels.index(p_id)][0] + 1
                    prev_prob = [i*occur for i in prev_prob_norm]                    
                    
                    if self.update_prob_method == "avg" or self.update_prob_method == "sum":
                        prob_values = evidence[counter]
                    elif self.update_prob_method == "evidence":
                        prob_values = ie.posterior(id_v)

                    total_prob = list(np.add(prev_prob, prob_values[:])) #faster than zip, map, np.sum
                    
                    norm_total_prob = self.normaliseSum(total_prob)

                    self.r_bn.cpt(id_v)[{"I": p_id}] = norm_total_prob[:]

                    self.cpt_matrix[p_id_index][counter] = norm_total_prob[:]
        
        # update the occurrences such that by multiplying the previous likelihoods, we can obtain the correct overall sum of the recognitions           
        if not self.isMultipleRecognitions:
            self.occurrences[self.i_labels.index(p_id)][0] += 1
            self.occurrences[self.i_labels.index(p_id)][2] += 1                    
        elif num_recog == num_mult_recognitions -1:
            self.occurrences[self.i_labels.index(p_id)][0] += 1
            self.occurrences[self.i_labels.index(p_id)][2] += num_mult_recognitions
    
    #---------------------------------------------FUNCTIONS TO SET SESSION CONSTANT/VARIABLES ---------------------------------------------# 

    def setSessionConstant(self, isMemoryRobot = True, isDBinCSV = False, isMultipleRecognitions = False, defNumMultRecog = 3, isSaveRecogFiles = True, isSpeak = True):
        """
        Set session constants: isMemoryRobot, isDBinCSV, isMultipleRecognitions, defNumMultRecog, isSaveRecogFiles, isSpeak from the file 
        isBNLoaded = False, isBNSaved = False, and sentences for recognition are loaded
        """
        self.isMemoryRobot = isMemoryRobot
        self.isDBinCSV = isDBinCSV
        self.isMultipleRecognitions = isMultipleRecognitions
        self.def_num_mult_recognitions = defNumMultRecog
        self.isSpeak = isSpeak
        self.isSaveRecogFiles = isSaveRecogFiles
        self.loadSentencesForRecognition()
        self.isBNLoaded = False
        self.isBNSaved = False
    
    def setSessionVar(self, isRegistered = True, isAddPersonToDB = False, personToAdd = []):
        """Set session variables: isRegistered, isAddPersonToDB, personToAdd."""
        self.start_recog_time = time.time()
        self.isRegistered = isRegistered
        self.isAddPersonToDB = isAddPersonToDB
        self.personToAdd = personToAdd
        self.image_id = None
        self.num_mult_recognitions = self.def_num_mult_recognitions
        if not self.isMemoryOnRobot:
            textToSay = self.lookAtTablet
            if self.isMemoryRobot and self.isRegistered:
                textToSay += self.pleasePhrase
            else:
                textToSay += self.enterName
            if self.isSpeak:
                self.say(textToSay)
                
    def setPersonToAdd(self, personToAdd):
        """Set person to add (used if the person is enrolled through the robot)"""
        self.isAddPersonToDB = True
        self.personToAdd = personToAdd
        
    #---------------------------------------------FUNCTIONS TO START RECOGNITION AND CONFIRM RECOGNITION (TWO MAIN FUNCTIONS NECESSARY TO TEST THE SYSTEM!!!)---------------------------------------------# 
        
    def startRecognition(self, recog_results_from_file = None):
        """
        Calls recognise method and says the identity estimated for confirmation (if isSpeak True). Returns estimated identity ID
        IMPORTANT: call setSessionConstant and setSessionVar and take picture before calling this function
        """
        identity_est = self.recognise(isRegistered = self.isRegistered, recog_results_from_file = recog_results_from_file)
        
        if self.isMemoryRobot and self.isRegistered and not self.isMemoryOnRobot:
            if identity_est == "":
                textToSay = self.noFaceInImage
                # TODO: take another picture from tablet
                if recog_results_from_file is None and self.isSpeak:
                    self.say(textToSay)
                # self.startRecognition(recog_results_from_file = recog_results_from_file)
            elif identity_est == self.unknown_var:
                textToSay = self.unknownPerson
            else:
                identity_say = self.names[self.i_labels.index(identity_est)].split()
                textToSay = self.askForIdentityConfirmal.replace("XX", str(identity_say[0]))
            if recog_results_from_file is None and self.isSpeak:
                self.say(textToSay)
#            print textToSay
        self.identity_est = identity_est
        return identity_est
    
    def confirmPersonIdentity(self, p_id = None, recog_results_from_file = None, isRobotLearning = True):
        """
        After the user confirms or enters the identity, the information is fed back to the system for updating the parameters.
        Calls setPersonIdentity method. If isSpeak, the system will give feedback depending if the user is correctly recognised or not.
        Comparison file is saved at this stage.
        IMPORTANT: call startRecognition before calling this function, and then ask for name from the person"""
        c_time_t = time.time()
        self.saveFilesToLastSaved() # save the current files to LastSaved folder (to recover in case of erroneous recognitions)
        name = self.setPersonIdentity(isRegistered = self.isRegistered, p_id = p_id, recog_results_from_file = recog_results_from_file, isRobotLearning=isRobotLearning)
        
        if self.isMemoryRobot:
            if name == "":
                textToSay = self.noFaceInImage
                if recog_results_from_file is None and self.isSpeak:
                    self.say(textToSay)
                self.confirmPersonIdentity(p_id = p_id, recog_results_from_file = recog_results_from_file, isRobotLearning=isRobotLearning)
            identity_say = name.split()
            if p_id is not None:
                if self.isRegistered:
                    falseRecognitionSentence = random.choice(self.falseRecognition)
                    textToSay = falseRecognitionSentence.replace("XX", str(identity_say[0]))
                else:
                    if self.userAlreadyRegistered:
                        textToSay = self.falseRegistration.replace("XX", str(identity_say[0]))
                    else:
                        textToSay = self.registrationPhrase.replace("XX", str(identity_say[0]))
            else:
                correctRecognition = random.choice(self.correctRecognition)
                textToSay = correctRecognition.replace("XX", str(identity_say[0]))
            
            if recog_results_from_file is None and self.isSpeak:
                self.say(textToSay)
                
        calc_time = time.time() - self.start_recog_time
        if p_id is None:
            identity_real = self.identity_est
        else:
            identity_real = p_id
        time_before_save = time.time()
        
        if self.isSaveRecogFiles:
            self.saveComparisonCSV(self.comparison_file, identity_real, self.identity_est, self.face_est, self.identity_est_prob, self.face_prob, calc_time, self.quality_estimate)
        self.num_recognitions += 1
        if self.isDebugMode:
            print "time to save comparison file: " + str(time.time() - time_before_save)    
    
    #---------------------------------------------FUNCTIONS FOR RECOGNITION---------------------------------------------# 

    def recognise(self, isRegistered = True, recog_results_from_file = None):
        """
        Recognise the user using the network:
        (1) Load BN if not already loaded
        (2) Get recognition results from modalities
        (3) Set evidence
        (4) Estimate identity using the network
        (5) Estimate identity using face recognition (for comparison)
        isRegistered = False if register button is pressed"""
        self.recog_results = []
        self.recog_results_from_file = recog_results_from_file
        self.num_mult_recognitions = self.def_num_mult_recognitions
        r_time_t = time.time()
        if not self.isBNLoaded:
            # (1)
            self.loadBN(self.recog_file, self.recogniser_csv_file, self.initial_recognition_file)
            if self.isDebugMode:
                print "time to load bn:" + str(time.time() - r_time_t)
        
        # (2)
        if self.isMultipleRecognitions:
            self.mult_recognitions_list = []
            self.recog_results_list = []
            self.ie_list = []
            self.discarded_data = []
            self.identity_est_list = []
            self.evidence_list = []
            if self.recog_results_from_file is None:
                # do parallel
                p_start_time = time.time()
                pool = ThreadPool(self.num_mult_recognitions)
                joint_results = pool.map(self.threadedRecognisePerson, [i for i in range(0, self.num_mult_recognitions)])
                pool.close()
                pool.join()
                num_rr = 0
                while num_rr < self.num_mult_recognitions:
                    # if there is no face in the image, discard it from the results
                    
                    if not joint_results[num_rr]:
                        joint_results.pop(num_rr)
                        self.discarded_data.append(num_rr)
                        self.num_mult_recognitions -= 1
                    else:
                        num_rr += 1
                        
                if self.num_mult_recognitions == 0:
                    print "Images are all discarded. No face detected in the images"
                    self.num_mult_recognitions = self.def_num_mult_recognitions
                    return ""
                    
                self.recog_results_list = [i[0] for i in joint_results]
                self.mult_recognitions_list = [i[1] for i in joint_results]
                self.ie_list = [i[2] for i in joint_results] 
                self.identity_est_list = [i[3] for i in joint_results]
                self.evidence_list = [i[4] for i in joint_results]
                p_end_time = time.time()
                if self.isDebugMode: 
                    print "time for parallel recognition: " + str(p_end_time - p_start_time)
            else:
                #do sequential
                for num_recog in range(0, self.num_mult_recognitions):
                    self.recog_results = self.recognisePerson(num_recog)
                    if not self.recog_results:
                        # if there is no face in the image, discard it from the results
                        self.discarded_data.append(num_recog)
                        continue
                    self.mult_recognitions_list.append(self.recog_results)
                    self.recog_results_list.append(self.recog_results)
                    if self.num_people > 1:
                        self.ie, evidence = self.setEvidence(self.recog_results) # (3)
                        self.ie_list.append(self.ie)
                        self.evidence_list.append(evidence)
                        i_post = np.array(self.ie.posterior(self.I)[:])
                        self.identity_est, self.quality_estimate = self.getEstimatedIdentity(i_post) # (4)
                    else:
                        self.identity_est, self.quality_estimate = self.getEstimatedIdentity() # (4)
                    self.identity_est_list.append(self.identity_est)
                self.num_mult_recognitions -= len(self.discarded_data)
                if self.num_mult_recognitions == 0:
                    print "Images are all discarded. No face detected in the images"
                    self.num_mult_recognitions = self.def_num_mult_recognitions
                    return ""
                    
            if self.num_people > 1:
                for r in range(0, self.num_mult_recognitions):
                    if r== 0:
                        ie_avg = self.ie_list[r].posterior(self.I)[:]
                    else:
                        temp_p = self.ie_list[r].posterior(self.I)[:]
                        ie_avg = [x + y for x, y in zip(ie_avg, temp_p)]
                identity_est_prob = self.normaliseSum(ie_avg)
                self.identity_prob_list = [float("{0:.4f}".format(i)) for i in identity_est_prob]
                self.identity_est, self.quality_estimate = self.getEstimatedIdentity(self.identity_prob_list) # (4)
                if self.isDebugMode:
                    print "self.identity_est:" + str(self.identity_est)
            else:
                self.identity_est, self.quality_estimate = self.getEstimatedIdentity() # (4)
                self.identity_prob_list = [1.0] # for unknown
        else:
            self.recog_results = self.recognisePerson() # (2)
            if not self.recog_results:
                print "No face detected in the image"
                self.num_mult_recognitions = self.def_num_mult_recognitions
                return ""
            self.nonweighted_evidence = self.recog_results
            if self.num_people > 1:
                self.evidence_list = []
                self.ie, evidence = self.setEvidence(self.recog_results) # (3)
                self.evidence_list.append(evidence)
                identity_est_prob = np.array(self.ie.posterior(self.I)[:])
                self.identity_prob_list = [float("{0:.4f}".format(i)) for i in identity_est_prob]
                self.identity_est, self.quality_estimate = self.getEstimatedIdentity(self.identity_prob_list) # (4)
            else:
                self.identity_est, self.quality_estimate = self.getEstimatedIdentity() # (4)
                self.identity_prob_list = [1.0] # for unknown
        if self.isDebugMode:
            print "time for recognise: " + str(time.time() - r_time_t)
        self.identity_est_prob = self.identity_prob_list
        self.face_est, self.face_prob = self.getFaceRecogEstimate() # (5)
        return self.identity_est

    def threadedRecognisePerson(self, num_recog):
        """Threaded recognisePerson with estimation of the identity (during recognise for multipleRecognitions)"""
        recog_results = self.recognisePerson(num_recog)
        if not recog_results:
            return []
        mult_recognitions = recog_results
        ie = None
        evidence = None
        if self.num_people > 1:
            ie, evidence = self.setEvidence(recog_results)
            i_post = np.array(ie.posterior(self.I)[:])
            identity_est, quality_estimate = self.getEstimatedIdentity(i_post)
        else:
            identity_est, quality_estimate = self.getEstimatedIdentity()
        
        return [recog_results, mult_recognitions, ie, identity_est, evidence]


    def threadedNoEstRecognisePerson(self, num_recog):
        """Threaded recognisePerson without estimation of the identity (during setPersonIdentity for multipleRecognitions)"""
        recog_results = self.recognisePerson(num_recog)
        mult_recognitions = recog_results
        ie = None
        if self.num_people > 1:
            ie, evidence = self.setEvidence(recog_results)
        return [recog_results, mult_recognitions, ie, evidence]


    #---------------------------------------------FUNCTIONS FOR SETTING THE IDENTITY OF THE PERSON AFTER CONFIRMATION---------------------------------------------# 


    def setPersonIdentity(self, isRegistered = True, p_id = None, recog_results_from_file = None, isRobotLearning = True):
        """
        This method saves the information from the recognise function with the actual identity of the person:
        (1) Initial recognition values are saved
        (2) The image is learned in face recognition db
        if new user:
            (3) Online learning (updateProbabilities method) for unknown state
            (4) Analysis file is saved (if isLogMode True)
            (5) New user is added to the DB and the BN
            (6) Recognition is made again (after the user is added to the face database)
        (7) Save images
        (8) Online learning (updateProbabilities method)
        (9) Save recognition files
        (10) Update BN
        (11) Update DB
        """
        self.recog_results_from_file = recog_results_from_file
        self.discarded_data = []
        isPrevSavedToAnalysis = False
        start_set_person = time.time()
        
        if self.isSaveRecogFiles:
            # (1) initial recognition file is saved here, because in the case that the recognition was ignored (either due to wrong input from the user
            # or because of timeout), the data will not be written unless there is a confirmation.
            if self.isMultipleRecognitions:
                for num_recog in range(0, self.num_mult_recognitions):
                    self.saveInitialRecognitionCSV(self.initial_recognition_file, self.recog_results_list[num_recog], self.identity_est_list[num_recog])
            else:
                 self.saveInitialRecognitionCSV(self.initial_recognition_file, self.recog_results, self.identity_est)

        end_save_init_recog_time = time.time()
        if self.isDebugMode:
            print "time to save initial recognition:" + str(end_save_init_recog_time-start_set_person)
        if p_id is None:
            p_id = self.identity_est
        if self.isAlreadyRegistered(p_id):
            self.userAlreadyRegistered = True
            if not isRegistered:
                logging.debug("The user is already registered.")
        else:
            self.userAlreadyRegistered = False
            if isRegistered:
                isRegistered = False
                self.isRegistered = False
        if self.isDebugMode:
            print "time to check if user is in db: " + str(time.time()-end_save_init_recog_time)
             
        if not isRegistered:
            # NOTE: make sure previous images are saved here! (If the images are not saved during enrollment by the robot, call saveImageAfterRecognition(isRegistered, p_id) to save them here)
            
            if self.userAlreadyRegistered:
                # (2)
                self.learnPerson(self.userAlreadyRegistered, p_id, isRobotLearning)
                if self.isDebugMode:
                    print "time to learn:" + str(time.time() - end_check_user_registered)
            else:
                if self.num_people > 1 and self.update_prob_unknown_method == "evidence":
                    # (3)
                    if self.isMultipleRecognitions:
                        for num_recog in range(0, self.num_mult_recognitions):
                            self.updateProbabilities(self.unknown_var, self.ie_list[num_recog], self.evidence_list[num_recog], num_recog)
                    else:
                        self.updateProbabilities(self.unknown_var, self.ie, self.evidence_list[0])  
                        
                time_after_update_unknown = time.time()
                self.learnPerson(self.userAlreadyRegistered, p_id, isRobotLearning)
                if self.isDebugMode:
                    print "time to learn:" + str(time.time() - time_after_update_unknown)
            
            if self.num_people > 1:
                # (4)
                start_save_analysis_time = time.time()
                if self.isLogMode and self.isSaveRecogFiles:
                    self.analysis_data_list = []
                    if self.isMultipleRecognitions:
                        for num_recog in range(0, self.num_mult_recognitions):
                            self.saveAnalysisFile(self.recog_results_list[num_recog], p_id, self.ie_list[num_recog], isPrevSavedToAnalysis, num_recog = num_recog)
                    else:
                        self.saveAnalysisFile(self.recog_results, p_id, self.ie, isPrevSavedToAnalysis)
                isPrevSavedToAnalysis = True
                if self.isDebugMode:
                    print "save analysis to db time: " + str(time.time() - start_save_analysis_time)
            
            if self.isAddPersonToDB:
                # (5)
                start_add_person_time = time.time()
                self.addPersonToBN(self.personToAdd)
                time.sleep(0.1)
                if self.isDebugMode:
                    print "time to add person" + str(time.time() - start_add_person_time)
            
            self.num_mult_recognitions = self.def_num_mult_recognitions
            start_recognise_time = time.time()
            
            # (6)
            if self.isMultipleRecognitions:
                if recog_results_from_file is None:
                    # PARALLEL
                    p_start_time = time.time()
                    pool = ThreadPool(self.num_mult_recognitions)
                    joint_results = pool.map(self.threadedNoEstRecognisePerson, [i for i in range(0, self.num_mult_recognitions)])
                    pool.close()
                    pool.join()
                    num_rr = 0
                    while num_rr < self.num_mult_recognitions:
                        # if there is no face in the image, discard it from the results
                        
                        if not joint_results[num_rr]:
                            joint_results.pop(num_rr)
                            self.discarded_data.append(num_rr)
                            self.num_mult_recognitions -= 1
                        else:
                            num_rr += 1
    
                    if self.num_mult_recognitions == 0:
                        print "Images are all discarded. No face detected in the images"
                        self.num_mult_recognitions = self.def_num_mult_recognitions
                        return ""
                    
                    self.recog_results_list = [i[0] for i in joint_results]
                    self.mult_recognitions_list = [i[1] for i in joint_results]
                    self.ie_list = [i[2] for i in joint_results]
                    self.evidence_list = [i[3] for i in joint_results]
                else:
                    # SEQUENTIAL
                    self.mult_recognitions_list = []
                    self.recog_results_list = []
                    for num_recog in range(0, self.num_mult_recognitions):
                        self.recog_results = self.recognisePerson(num_recog = num_recog)
                        if not self.recog_results:
                            # if there is no face in the image, discard it from the results
                            self.discarded_data.append(num_recog)
                            continue
                        self.mult_recognitions_list.append(self.recog_results) 
                        self.recog_results_list.append(self.recog_results)
                    self.num_mult_recognitions -= len(self.discarded_data)
                    
                    if self.num_mult_recognitions == 0:
                        print "Images are all discarded. No face detected in the images"
                        self.num_mult_recognitions = self.def_num_mult_recognitions
                        return ""
                
            else:
                self.recog_results = self.recognisePerson()
                if not self.recog_results:
                    self.recog_results = self.recognisePerson() # try again, ALFaceDetection is unreliable, it doesn't work sometimes. It is good to run it again to make sure it doesn't recognise a face
                    if not self.recog_results:
                        print "Image is discarded. No face detected in the image"
                        self.num_mult_recognitions = self.def_num_mult_recognitions
                        return ""
            if self.isDebugMode:
                print "time to recognise for registering:" + str(time.time() - start_recognise_time)

                    
        else:
            start_learn_time = time.time()
            self.learnPerson(isRegistered, p_id, isRobotLearning)
            if self.isDebugMode:
                print "time to learn (add picture):" + str(time.time() - start_learn_time)
        
        start_image_save_time = time.time()
        
        # (7)
        if self.isSaveRecogFiles:
            self.saveImageAfterRecognition(isRegistered, p_id)
        if self.isDebugMode:
            print "time to save images for recognition:" + str(time.time() - start_image_save_time)
        
        if self.num_people < 2 and self.isSaveRecogFiles:
            # (9)
            if self.isMultipleRecognitions:
                for num_recog in range(0, self.num_mult_recognitions):
                    self.nonweighted_evidence = self.mult_recognitions_list[num_recog]
                    self.saveRecogniserCSV(self.recogniser_csv_file, p_id, num_recog = num_recog)
            else:
                self.nonweighted_evidence = self.recog_results
                self.saveRecogniserCSV(self.recogniser_csv_file, p_id)
        else:
            if not isRegistered:
                # get the inference from the final recognition
                self.evidence_list = []
                start_posterior_calc = time.time()
                if self.isMultipleRecognitions:
                    if recog_results_from_file is not None:
                        self.ie_list = []
                        for num_recog in range(0, self.num_mult_recognitions):
                            self.ie, evidence = self.setEvidence(self.recog_results_list[num_recog])
                            self.ie_list.append(self.ie)
                            self.evidence_list.append(evidence)
                    
                        if self.isDebugMode:
                            print "time to calculate posterior after registering:" + str(time.time()- start_posterior_calc)
                    
                    start_time_avg_posterior = time.time()
                    for num_recog in range(0, self.num_mult_recognitions):
                        if num_recog== 0:
                            ie_avg = self.ie_list[num_recog].posterior(self.I)[:]
                        else:
                            temp_p = self.ie_list[num_recog].posterior(self.I)[:]
                            ie_avg = [x + y for x, y in zip(ie_avg, temp_p)]

                    if self.isDebugMode:
                        print "time to avg posterior:" + str(time.time() - start_time_avg_posterior)
                        
                    identity_est_prob = self.normaliseSum(ie_avg)
                else:
                    self.ie, evidence = self.setEvidence(self.recog_results)
                    self.evidence_list.append(evidence)
                    identity_est_prob = np.array(self.ie.posterior(self.I)[:])
                
                self.identity_prob_list = [float("{0:.4f}".format(i)) for i in identity_est_prob]       
             
            if self.isMultipleRecognitions:
                self.analysis_data_list = []
                for num_recog in range(0, self.num_mult_recognitions):
                    self.nonweighted_evidence = self.mult_recognitions_list[num_recog]
                    self.recog_results = self.recog_results_list[num_recog]
                    self.ie = self.ie_list[num_recog]
                    
                    self.updateProbabilities(p_id, self.ie, self.evidence_list[num_recog]) # (8)
                    if self.isSaveRecogFiles:
                        if not self.isBNSaved:
                            self.saveBN(num_recog = num_recog) # (10)
                        
                        self.saveRecogniserCSV(self.recogniser_csv_file, p_id, num_recog = num_recog) # (9)
                        self.updateDB(self.db_file, p_id) # (11)
                        if self.isLogMode:
                            self.saveAnalysisFile(self.recog_results, p_id, self.ie, isPrevSavedToAnalysis, num_recog = num_recog) # (9)
            else:
                self.nonweighted_evidence = self.recog_results
                self.updateProbabilities(p_id, self.ie, self.evidence_list[0]) # (8)
                
                if self.isSaveRecogFiles:
                    if not self.isBNSaved:
                        self.saveBN() # (10)
                    self.saveRecogniserCSV(self.recogniser_csv_file, p_id) # (9)
                    self.updateDB(self.db_file, p_id) # (11)
                    if self.isLogMode:
                        self.saveAnalysisFile(self.recog_results, p_id, self.ie, isPrevSavedToAnalysis) # (9)

        return self.names[self.i_labels.index(p_id)]

    #---------------------------------------------FUNCTIONS FOR GETTING/SETTING EVIDENCE FROM THE RECOGNITION RESULTS (MULTI-MODALITIES)---------------------------------------------# 


    def setEvidence(self, recog_results, param_weights = None):
        """Calculate the evidence for each parameter from the recognition results in terms of probabilities, and set the evidence of the network"""
        # self.printPriors()
        
        if param_weights is None:
            param_weights = self.weights
        
        start_time_set_ev1 = time.time()
        # P(e|F)
        face_result = self.setFaceProbabilities(recog_results[0], param_weights[0])
        
        # P(e|G)
        gender_result = self.setGenderProbabilities(recog_results[1], param_weights[1])

        # P(e|A)
        age_result = self.getCurve(conf = recog_results[2][1], mean = recog_results[2][0], min_value = self.age_min, max_value = self.age_max, weight = param_weights[2], norm_method = self.evidence_norm_methods[2])

        # P(e|H)
        height_result = self.getCurve(conf = recog_results[3][1], mean = recog_results[3][0], stddev = self.stddev_height, min_value = self.height_min, max_value = self.height_max, weight = param_weights[3], norm_method = self.evidence_norm_methods[3])

        # P(e|T)   
        time_result = self.getCurve(mean = self.getTimeSlot(recog_results[4]), stddev = self.stddev_time, min_value = self.time_min, max_value = self.time_max, weight = param_weights[4], norm_method = self.evidence_norm_methods[4])
        
#         gnb.showInference(self.r_bn,evs={"F":face_result, "G":gender_result, "A":age_result, "H":height_result, "T":time_result})
        evidence = [face_result, gender_result, age_result, height_result, time_result]

        ie = gum.LazyPropagation(self.r_bn)
        ie.setEvidence({"F":face_result, "G":gender_result, "A":age_result, "H":height_result, "T":time_result})
        ie.makeInference()

        return ie, evidence
                        
    def setFaceProbabilities(self, face_values, weight, isNormalisationOn = True):
        """
        Set face probabilities for evidence using the face similarity scores:
        Similarity scores are sorted according to the order in the i_labels,
        face_recognition_threshold is added for the unknown state,
        weight is applied to all values,
        and values are normalised
        """
        accuracy_face = face_values[0]
        face_similarities = []
        if len(face_values[1]) > 0:
            face_similarities = face_values[1][:]
        if len(face_similarities) == 0:
            face_similarities.append([self.unknown_var, 1.0])
        elif not (self.unknown_var in (x[0] for x in face_similarities)):
#             max_similarity = max(face_similarities, key=lambda x: x[1])[1] # maximum similarity score in the face recognition
#             face_similarities.append([self.unknown_var, 1.0 - max_similarity]) # the similarity score of unknown is 1-max_similarity # don't use, decreases recognition rate
            face_similarities.append([self.unknown_var, self.face_recog_threshold])
#             face_similarities.append([self.unknown_var, self.getFaceThreshold(self.face_recog_threshold)]) # don't use, decreases recognition rate

        r_results_names = []
        for counter in range(0, len(face_similarities)):
            r_results_names.append(face_similarities[counter][0])
        
        r_results_index = []
        for counter in range(0, len(self.i_labels)):
            if self.i_labels[counter] in r_results_names:
                r_results_index.append(r_results_names.index(self.i_labels[counter]))
            else:
                # if the person in database is not in face recognition database yet (did not have his/her first session yet)
                r_results_index.append(-1)

        face_result = []
        for counter in range(0, len(self.i_labels)):
            # values are normalised when using this method 
            if r_results_index[counter] == -1:
                fr = self.prob_threshold
            else:
                fr = face_similarities[r_results_index[counter]][1]
                if fr < self.prob_threshold:
                    fr = self.prob_threshold
            accur = self.applyFaceAccuracy(fr, accuracy_face)
            face_result.append(self.applyWeight(accur, weight))

        if isNormalisationOn:
            face_result = self.normalise(face_result, norm_method = self.evidence_norm_methods[0])
        return face_result

    def setGenderProbabilities(self, gender_values, weight):
        """Set gender probabilities by applying the weight. P(G='Female'|I=i) = 1 - P(G='Male'|I=i) """
        gr = gender_values[1]

        if gr > 1.0 - self.prob_threshold:
            gr = 1 - self.prob_threshold    
        gr_comp = 1 - gr
        
        gr_comp = self.applyWeight(gr_comp, weight)
        gr = self.applyWeight(gr, weight)
        sum_gr = gr + gr_comp
        if sum_gr == 0:
            gender_result = [0.5, 0.5]
        else:
            if gender_values[0] == self.g_labels[0]:
                gender_result = [gr/sum_gr, gr_comp/sum_gr] 
            else:
                gender_result = [gr_comp/sum_gr, gr/sum_gr]
        return gender_result
    
    def getNonweightedProbabilities(self):
        """The evidence (recognition probabilities of F,G,A,H,T) without weights applied"""
        
        # P(e|F)
        face_result = self.setFaceProbabilities(self.recog_results[0], 1.0)
        
        # P(e|G)
        gender_result = self.setGenderProbabilities(self.recog_results[1], 1.0)

        # P(e|A)
        age_result = self.getCurve(conf = self.recog_results[2][1], mean = self.recog_results[2][0], min_value = self.age_min, max_value = self.age_max, weight = 1.0, norm_method = self.evidence_norm_methods[2])

        # P(e|H)
        height_result = self.getCurve(conf = self.recog_results[3][1], mean = self.recog_results[3][0], stddev = self.stddev_height, min_value = self.height_min, max_value = self.height_max, weight = 1.0, norm_method = self.evidence_norm_methods[3])
        
        # P(e|T)   
        time_result = self.getCurve(mean = self.getTimeSlot(self.recog_results[4]), stddev = self.stddev_time, min_value = self.time_min, max_value = self.time_max, weight = 1.0, norm_method = self.evidence_norm_methods[4])
    
        return [face_result, gender_result, age_result, height_result, time_result] 

    
    def getPosteriorIUsingCalculatedEvidence(self, bn, evidence):
        """Get the posterior of identity node using the current evidence"""
        ie = gum.LazyPropagation(bn)
        ie.setEvidence({"F":evidence[0], "G":evidence[1], "A":evidence[2], "H":evidence[3], "T":evidence[4]})
        ie.makeInference()
        post_I = ie.posterior(self.I)[:]
        return post_I
    
    def getEstimatedProbabilities(self):
        """Get estimated probabilities"""
        if self.num_people > 1:
            est_prob = self.identity_prob_list
        else:
            est_prob = [1.0] # unknown
        return est_prob
        
    #---------------------------------------------CALCULATION FUNCTIONS---------------------------------------------#
    
    def uniformDistribution(self, min_value, max_value):
        """Returns a list of probability values with equal values: 1/(max_value - min_value + 1) . 
        Min_value and max_value specify how many states are there in the list. 
        e.g. for height, min_height: 50, max_height: 240. There are 191 states. 
        If each height is equally likely, then the probability for each height is 1/191."""
        uni_value = 1.0/(max_value - min_value + 1)
        return [uni_value for x in range(min_value, max_value + 1)]
    
    def getCurve(self, conf = 1.0, mean = 0.0, stddev = 0.0, min_value = 0, max_value = 0, weight = 1.0, norm_method = None):
        """Get discretised and normalised normal curve for estimations of each state in a node (for range parameters: A, H and T)"""
        curve = []
        if conf > self.max_threshold:
            conf = self.max_threshold # decrease the prob. to get a Gaussian distribution
            
        if np.isclose(stddev, 0.0) and conf >= self.conf_threshold:
            # applicable to age only
            stddev = 0.5/self.normppf(conf + (1-conf)/2.0)
        
        if conf < self.conf_threshold or weight == 0.0:
            # uniform distribution
            curve = self.uniformDistribution(min_value, max_value)
        else:
            # Gaussian distribution         
#             norm_curve = norm(loc=observed_height,scale=self.stddev_height )
            for j in range(min_value, max_value +1):
#                 j_pdf = norm_curve.pdf(j)
                j_pdf = self.normpdf(j, mean, stddev)
                if j_pdf < self.prob_threshold:
                    j_pdf = self.prob_threshold
                curve.append(self.applyWeight(j_pdf, weight))
            curve = self.normalise(curve, norm_method)

        return curve
    
#     def getFaceThreshold(self, accuracy):
#         """"""
#         # NOTE: DON'T USE, MAKES RECOGNITION WORSE
#         return self.face_max_threshold - ((self.face_max_threshold- self.face_min_threshold)*accuracy/1.0)

    def getEstimatedIdentity(self, i_post = None):
        """Get estimated identity from the highest posterior probability (=argmax P(I|F,G,A,H,T)) that is greater than the quality threshold"""
        
        identity_est = ""
        self.isUnknownCondition = False
        if self.num_people > 1 and self.num_recognitions >= self.num_recog_min:
            identity_est = self.i_labels[np.argmax(i_post)]
            quality_estimate = self.getQualityEstimation(i_post)     
            if quality_estimate < self.quality_threshold or quality_estimate == 0:
                # if all states are equally likely or if the quality lower than quality_threshold then person is unknown
                self.isUnknownCondition = True
                identity_est = self.unknown_var
                if self.isDebugMode:            
                    print "unknown condition (quality < quality_threshold or quality == 0)"
 
        else:
            identity_est = self.unknown_var
            quality_estimate = -1
        identity_est = str(identity_est)
        return identity_est, quality_estimate
    
    def getQualityEstimation(self, i_post = None):
        """Quality = (highest_prob - second_highest_prob) * num_people"""
        quality = -1.0
        if self.num_people > 1 and i_post is not None:
            two_largest = heapq.nlargest(2, i_post)
            if self.qualityCoefficient is None:
#                 quality = (two_largest[0] - two_largest[1]) * self.num_recognitions
                quality = (two_largest[0] - two_largest[1]) * self.num_people

            else:
                quality = (two_largest[0] - two_largest[1]) * self.qualityCoefficient
        return float("{0:.5f}".format(quality))
    
    def getFaceRecognitionValues(self, face_values):
        """Sort similarity scores in the order of i_labels"""
        accuracy_face = face_values[0]
        face_similarities = []
        if len(face_values[1]) > 0:
            face_similarities = face_values[1][:]
            
        r_results_names = []
        for counter in range(0, len(face_similarities)):
            r_results_names.append(face_similarities[counter][0])
        
        r_results_index = []
        # exclude unknown_var -> start from 1
        for counter in range(1, len(self.i_labels)):
            if self.i_labels[counter] in r_results_names:
                r_results_index.append(r_results_names.index(self.i_labels[counter]))
            else:
                # if the person in database is not in face recognition database yet (did not have his/her first session yet)
                r_results_index.append(-1)
        face_prob = []
        for r_counter in range(0, len(r_results_index)):
            if r_results_index[r_counter] != -1:
#                 face_prob.append(self.applyFaceAccuracy(face_similarities[r_results_index[r_counter]][1], accuracy_face))
                face_prob.append(face_similarities[r_results_index[r_counter]][1])
            else:
                face_prob.append(0.0) # for face recognition, it is 0 NOT self.prob_threshold
        return face_prob # doesnt include unknown!!!
        
    def getFaceRecogEstimate(self):
        """Get identity estimated by face recognition"""
        total_face_prob = []
            
        if self.isMultipleRecognitions:
            for num_recog in range(0, self.num_mult_recognitions):                
                face_prob = self.getFaceRecognitionValues(self.mult_recognitions_list[num_recog][0])
                if num_recog == 0:
                    total_face_prob = face_prob[:]
                else:
                    total_face_prob = [x + y for x, y in zip(total_face_prob, face_prob)]
            total_face_prob = [i/(self.num_mult_recognitions*1.0) for i in total_face_prob]
        else:
            total_face_prob = self.getFaceRecognitionValues(self.nonweighted_evidence[0])
        unknown_index = self.i_labels.index(self.unknown_var)
        if not total_face_prob:
            return self.unknown_var, [self.face_recog_threshold]
        elif self.num_recognitions < self.num_recog_min:
            total_face_prob.insert(unknown_index, self.face_recog_threshold)
            return self.unknown_var, total_face_prob
        max_est_value = max(total_face_prob)
        max_est_identity = self.i_labels[1 + total_face_prob.index(max_est_value)]
        
        if max_est_value < self.face_recog_threshold:
            total_face_prob.insert(unknown_index, self.face_recog_threshold)
            return self.unknown_var, total_face_prob
                   
        total_face_prob.insert(unknown_index, 0.0)
        isclose_ar = np.isclose(total_face_prob, max_est_value)
        if len(isclose_ar[isclose_ar==True]) > 1:
            return self.unknown_var, total_face_prob
        return max_est_identity, total_face_prob

    def getAgeFromBirthYear(self, birth_year):
        """Calculate age given the year of birth"""
        age_p = date.today().year - birth_year
        if age_p > self.age_max:
            age_p = self.age_max
        return age_p

    
    def getTimeSlot(self, p_time):
        """Calculate time slot given time (time in format ['HH:MM:SS','1'] where '1' is the number of the day ('1' for Monday, '2' for Tuesday,..))"""
        tp = p_time[0].split(":")
        time_slot = (int(p_time[1])-1)*24*60/self.period + int(tp[0])*60/self.period + int(tp[1])/self.period
        return time_slot
    
    def applyWeight(self, value, weight):
        """Apply weight to the value: 
        pow (weight is taken as the power to the value - default), 
        invpow(1/weight is taken as the power to the value),
        mult (value is multiplied by the weight) - NOT USED since normalisation method can remove the effect"""
        if self.apply_weight_method == "pow":
            return math.pow(value, weight)
        elif self.apply_weight_method == "invpow":
            return math.pow(value, 1.0/weight)
        elif self.apply_weight_method == "mult":
            return value*weight
    
    def applyFaceAccuracy(self, value, accuracy):
        """Apply accuracy of face recognition to the value:
        none (not applied - default),
        pow (weight is taken as the power to the value), 
        invpow(1/weight is taken as the power to the value),
        mult (value is multiplied by the weight) - NOT USED since normalisation method can remove the effect
        """
        if self.apply_accuracy_method == "pow":
            return math.pow(value, accuracy)
        elif self.apply_accuracy_method == "invpow":
            return math.pow(value, 1.0/accuracy)
        elif self.apply_accuracy_method == "mult":
            return value*accuracy     
        elif self.apply_accuracy_method == "none":
            return value
        
    def normalise(self, array, norm_method = None):
        """Applies normalisation method to the given list:
        norm-sum: calls normaliseSum,
        softmax: calls softmax,
        minmax: calls minmax,
        tanh: calls tanhScore method"""
        if norm_method is None:
            norm_method = self.norm_method
        if norm_method == "norm-sum":
            return self.normaliseSum(array)
        elif norm_method == "softmax":
            return self.softmax(array)
        elif norm_method == "minmax":
            return self.minmax(array)
        elif norm_method == "tanh":
            return self.tanhScore(array)

    def normaliseSum(self, array):
        """Divide each value by the sum of values"""
        sum_array = sum(array)
        if sum_array == 0:
            return [1.0/len(array) for i in array]
        return [float(i) / sum_array for i in array]
        
    def softmax(self, array):
        """new_value=e^value/sum(e^value)"""
        array_exp = [math.exp(i) for i in array]  
        sum_array_exp = sum(array_exp)  
        return [i / sum_array_exp for i in array_exp]

    def minmax(self, array):
        """new_value= (value-min(array))/(max(array) - min(array))"""
        diff = (max(array) - min(array))*1.0
        min_ar = min(array)
        if diff > 0:
            return [(i-min_ar)/diff for i in array]
        len_ar = len(array)
        return [1.0/len_ar for i in array]

    def tanhScore(self, array):
        """new_value = 0.5*(np.tanh(0.01*((value - mean_a)/std_a))+1) """
        mean_a = np.mean(array)
        std_a = np.std(array)
        if std_a == 0:
            len_ar = len(array)
            return [1.0/len_ar for i in array]
        return [0.5*(np.tanh(0.01*((i - mean_a)/std_a))+1) for i in array]

    def getStddevFromConfidence(self, confidence):
        """From https://stats.stackexchange.com/questions/269784/calculating-the-probability-of-a-discrete-rv-given-the-mean-and-the-probability"""
        z = self.normppf(confidence + (1-confidence)/2.0) # z-score
        return 0.5/z
        
    def normpdf(self, x, loc=0, scale=1):
        """x is the value that pdf wants to be read at, loc is the mean, and scale is the stddev
        From: https://stackoverflow.com/questions/8669235/alternative-for-scipy-stats-norm-pdf"""
        u = float(x-loc) / abs(scale)
        y = np.exp(-u*u/2) / (np.sqrt(2*np.pi) * abs(scale))
        return y
    
    def normppf(self, y0):     
        """From https://stackoverflow.com/questions/41338539/how-to-calculate-a-normal-distribution-percent-point-function-in-python"""   
        
        s2pi = 2.50662827463100050242E0

        P0 = [
            -5.99633501014107895267E1,
            9.80010754185999661536E1,
            -5.66762857469070293439E1,
            1.39312609387279679503E1,
            -1.23916583867381258016E0,
        ]
        
        Q0 = [
            1,
            1.95448858338141759834E0,
            4.67627912898881538453E0,
            8.63602421390890590575E1,
            -2.25462687854119370527E2,
            2.00260212380060660359E2,
            -8.20372256168333339912E1,
            1.59056225126211695515E1,
            -1.18331621121330003142E0,
        ]
        
        P1 = [
            4.05544892305962419923E0,
            3.15251094599893866154E1,
            5.71628192246421288162E1,
            4.40805073893200834700E1,
            1.46849561928858024014E1,
            2.18663306850790267539E0,
            -1.40256079171354495875E-1,
            -3.50424626827848203418E-2,
            -8.57456785154685413611E-4,
        ]
        
        Q1 = [
            1,
            1.57799883256466749731E1,
            4.53907635128879210584E1,
            4.13172038254672030440E1,
            1.50425385692907503408E1,
            2.50464946208309415979E0,
            -1.42182922854787788574E-1,
            -3.80806407691578277194E-2,
            -9.33259480895457427372E-4,
        ]
        
        P2 = [
            3.23774891776946035970E0,
            6.91522889068984211695E0,
            3.93881025292474443415E0,
            1.33303460815807542389E0,
            2.01485389549179081538E-1,
            1.23716634817820021358E-2,
            3.01581553508235416007E-4,
            2.65806974686737550832E-6,
            6.23974539184983293730E-9,
        ]
        
        Q2 = [
            1,
            6.02427039364742014255E0,
            3.67983563856160859403E0,
            1.37702099489081330271E0,
            2.16236993594496635890E-1,
            1.34204006088543189037E-2,
            3.28014464682127739104E-4,
            2.89247864745380683936E-6,
            6.79019408009981274425E-9,
        ]
        if y0 <= 0 or y0 >= 1:
            raise ValueError("ndtri(x) needs 0 < x < 1")
        negate = True
        y = y0
        if y > 1.0 - 0.13533528323661269189:
            y = 1.0 - y
            negate = False
    
        if y > 0.13533528323661269189:
            y = y - 0.5
            y2 = y * y
            x = y + y * (y2 * self.polevl(y2, P0) / self.polevl(y2, Q0))
            x = x * s2pi
            return x
    
        x = math.sqrt(-2.0 * math.log(y))
        x0 = x - math.log(x) / x
    
        z = 1.0 / x
        if x < 8.0:
            x1 = z * self.polevl(z, P1) / self.polevl(z, Q1)
        else:
            x1 = z * self.polevl(z, P2) / self.polevl(z, Q2)
        x = x0 - x1
        if negate:
            x = -x
    
        return x

    def polevl(self, x, coef):
        """Polynomial evaluation, given x value and coefficient list"""
        accum = 0
        for c in coef:
            accum = x * accum + c
        return accum

    def isAlreadyRegistered(self, p_id):
        """Checks if the user is already registered: if user in i_labels and occurrence is greater than 1, returns True, else False"""
        if p_id in self.i_labels and self.occurrences[self.i_labels.index(p_id)][0] > 0:
            return True
        return False
    
    #---------------------------------------------FUNCTIONS FOR FILES---------------------------------------------# 
    
    def resetFilePaths(self):
        """Reset file paths to original values to the current directory. 
        The directory can be changed by calling setFilePaths(recog_folder)"""
        self.recog_file = "RecogniserBN.bif"
        self.recogniser_csv_file = "RecogniserBN.csv"
        self.initial_recognition_file = "InitialRecognition.csv"
        self.analysis_file = self.analysis_dir + "Analysis.json"
        self.comparison_file = self.analysis_dir + "Comparison.csv"
        self.db_file = "db.csv"
        self.stats_file = "stats.csv"
        self.conf_matrix_file = "confusionMatrix.csv"
        self.recog_folder = ""
        self.image_save_dir = "images/"
        self.previous_files_dir = "LastSaved/"
        self.faceDB = "faceDB"
                
    def setFilePaths(self, recog_folder):
        """Changes the directory to save the files"""
        self.recog_file = recog_folder + self.recog_file 
        self.recogniser_csv_file = recog_folder + self.recogniser_csv_file
        self.initial_recognition_file = recog_folder + self.initial_recognition_file
        self.analysis_file = recog_folder + self.analysis_file
        self.db_file = recog_folder + self.db_file
        self.comparison_file = recog_folder + self.comparison_file
        self.image_save_dir = recog_folder + "images/"
        self.stats_file = recog_folder + self.stats_file
        self.conf_matrix_file = recog_folder + self.conf_matrix_file
        self.previous_files_dir = recog_folder + self.previous_files_dir
        self.faceDB = recog_folder + self.faceDB
        self.recog_folder = recog_folder

    def resetFiles(self):
        """Removes the files/folders and rewrites empty files/folders."""
        if os.path.isfile(self.recog_file):
            os.remove(self.recog_file)
        if os.path.isfile(self.recogniser_csv_file):
            os.remove(self.recogniser_csv_file)
        with open(self.recogniser_csv_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["I", "F", "G", "A", "H", "T", "R", "N"])
        if os.path.isfile(self.initial_recognition_file):
            os.remove(self.initial_recognition_file)
        with open(self.initial_recognition_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["I_est", "F", "G", "A", "H", "T", "N"])
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)
        with open(self.db_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
#             writer.writerow(["id", "name", "gender", "age", "height", "times", "occurrence"])
            writer.writerow(["id", "name", "gender", "birthYear", "height", "times", "occurrence"])
        analys_dir = self.recog_folder + self.analysis_dir
        if os.path.isdir(analys_dir):
            shutil.rmtree(analys_dir)
        os.makedirs(analys_dir)        
        with open(self.comparison_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["I_real", "I_est", "F_est", "I_prob", "F_prob", "Calc_time", "R", "Quality", "Highest_I_prob", "Highest_F_prob"])
        if os.path.isdir(self.image_save_dir):
            shutil.rmtree(self.image_save_dir)
        os.makedirs(self.image_save_dir)
        os.makedirs(self.image_save_dir + "Known_True")
        os.makedirs(self.image_save_dir + "Known_False")
        os.makedirs(self.image_save_dir + "Known_Unknown")
        os.makedirs(self.image_save_dir + "Unknown_True")
        os.makedirs(self.image_save_dir + "Unknown_False")
        os.makedirs(self.image_save_dir + "discarded")
        
        if os.path.isdir(self.previous_files_dir):
            shutil.rmtree(self.previous_files_dir)
        os.makedirs(self.previous_files_dir)
        if os.path.isfile(self.faceDB):
            os.remove(self.faceDB)
            
    def saveFilesToLastSaved(self):
        # add files one by one to avoid delay (instead of transferring the entire folder)
        if os.path.isfile(self.recog_file):
            shutil.copy2(self.recog_file, self.previous_files_dir)
        if self.isSaveRecogFiles:
            if os.path.isfile(self.recogniser_csv_file):
                shutil.copy2(self.recogniser_csv_file, self.previous_files_dir)
            if os.path.isfile(self.initial_recognition_file):
                shutil.copy2(self.initial_recognition_file, self.previous_files_dir)
            if os.path.isfile(self.db_file):
                shutil.copy2(self.db_file, self.previous_files_dir)
            if os.path.isfile(self.stats_file):
                shutil.copy2(self.stats_file, self.previous_files_dir)
            if os.path.isfile(self.conf_matrix_file):
                shutil.copy2(self.conf_matrix_file, self.previous_files_dir)  
            analys_save_dir = self.previous_files_dir + self.analysis_dir
            if self.isLogMode and not os.path.isdir(analys_save_dir):
               os.makedirs(analys_save_dir)
            if self.isLogMode and os.path.isfile(self.comparison_file):
                shutil.copy2(self.comparison_file, analys_save_dir)
            if os.path.isfile(self.faceDB):
                shutil.copy2(self.faceDB, self.previous_files_dir)
    
    def revertToLastSaved(self, isRobot, num_recog):
        # overwrite the current files with the previously saved files (RecogniserBN.bif, db.csv, RecogniserBN.csv and InitialRecognition.csv)
        copy_tree(self.previous_files_dir, self.recog_folder)
        if isRobot:
            # if robot: set the previously saved faceDB as the current faceDB
            self.useFaceDetectionDB()

        if self.isLogMode and self.isSaveRecogFiles:
            # remove last analysis files
            analys_dir = self.recog_folder + self.analysis_dir 
            if os.path.isdir(analys_dir):
                 # remove last two analysis files
                analys_file = self.analysis_file.replace(".json","") + str(num_recog) + ".json"
                analys_file_2 = analys_file.replace(".json", "_2.json")
                if os.path.isfile(analys_file):
                    os.remove(analys_file)
                if os.path.isfile(analys_file_2):
                    os.remove(analys_file_2)
        for dirpath, dirnames, filenames in os.walk(self.image_save_dir):
            # remove image from folder
            for filename in filenames:
                if filename.startswith(str(num_recog) + "_"):
                    os.remove(str(dirpath)+"/"+filename)
                    break
        print "Reverted to the previous recognition files."
 
    def learnFromFile(self, db_list=None, init_list=None, recogs_list=None,
                            db_file = None, init_recog_file = None, final_recog_file = None, 
                            valid_info_file = None, isSaveImageAn = False, orig_image_dir = None):
        """Creates the network from files/lists. The evidence is fed from existing file or information one by one."""
        learn_start_time = time.time() 

        if db_list is None and os.path.isfile(db_file):
            try:
                df_db = pandas.read_csv(db_file, dtype={"id": object}, converters={"times": ast.literal_eval}, usecols = ["id","name","gender","birthYear","height","times"])
            except:
                df_db = pandas.read_csv(db_file, dtype={"id": object}, converters={"times": ast.literal_eval}, usecols = ["id","name","gender","age","height","times"])
            db_list = df_db.values.tolist()

        if init_list is None and os.path.isfile(init_recog_file):
            df_init = pandas.read_csv(init_recog_file, dtype={"I_est": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
            init_list = df_init.values.tolist()
            
        if recogs_list is None and os.path.isfile(final_recog_file):
            df_final = pandas.read_csv(final_recog_file, dtype={"I": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
            recogs_list = df_final.values.tolist()
            
        if isSaveImageAn:
#             df_info = pandas.read_csv(valid_info_file, usecols ={"Bin", "Bin_image", "Validation_image"}).values.tolist()          
            df_info = pandas.read_csv(valid_info_file, usecols ={"Original_image", "Validation_image"}).values.tolist()          

        num_unknown = 0
        stats_openSet = [0,0] #DIR, FAR
        stats_FR = [0,0]
        numNoFaceImages = 0
        count_recogs = 0
        num_recog = 0
        while count_recogs < len(recogs_list):
            idPerson = str(recogs_list[count_recogs][0])
            isRegistered = not recogs_list[count_recogs][6]
            isAddPersonToDB = recogs_list[count_recogs][6]
            numRecognition = recogs_list[count_recogs][7]
            person = []
            recog_results = []
            if isAddPersonToDB:    
                person = [x for x in db_list if str(x[0]) == idPerson][0]
                person[0] = str(person[0])
                person[1] = str(person[1])
                isRegistered = False
                
            self.setSessionVar(isRegistered = isRegistered, isAddPersonToDB = isAddPersonToDB, personToAdd = person)    
            
            if isRegistered:
                if self.isMultipleRecognitions:
                    recog_values = [x for x in recogs_list if x[7] == numRecognition]
                    num_mult_recognitions = len(recog_values)
                    self.setDefinedNumMultRecognitions(num_mult_recognitions)
                    for num_rec in range(0, num_mult_recognitions):
                        recog_results.append(recog_values[num_rec][1:6])
                        if num_rec < num_mult_recognitions - 1:
                            count_recogs += 1
                else:
                    recog_results = recogs_list[count_recogs][1:6]
            else:
                if self.isMultipleRecognitions:
                    init_recog_values = [x for x in init_list if x[6] == numRecognition]
                    num_mult_recognitions = len(init_recog_values)
                    self.setDefinedNumMultRecognitions(num_mult_recognitions)
                    for num_rec in range(0, num_mult_recognitions):
                        recog_results.append(init_recog_values[num_rec][1:6])
                else:
                    init_recog_values = [x for x in init_list if x[6] == numRecognition][0]
                    recog_results = init_recog_values[1:6]
                                                               
            identity_est = self.startRecognition(recog_results) # get the estimated identity from the recognition network
            
            if identity_est == "":
                numNoFaceImages += 1
                continue
            p_id = None
            
            stats_openSet = self.getPerformanceMetrics(identity_est, idPerson, self.unknown_var, isRegistered, stats_openSet)
            
            stats_FR = self.getPerformanceMetrics(self.face_est, idPerson, self.unknown_var, isRegistered, stats_FR)
            
            isRecognitionCorrect = False

            if isRegistered:
                if identity_est != self.unknown_var:
                    if identity_est == idPerson:
                        isRecognitionCorrect = True # True if the name is confirmed by the user
                        
            if isSaveImageAn:
                copy_dir = ""
                if isRegistered:
                    if isRecognitionCorrect:
                        copy_dir = "Known_True/"
                    elif identity_est == self.unknown_var:
                        copy_dir = "Known_Unknown/"
                    else:
                        copy_dir = "Known_False/"
                else:
                    if identity_est == self.unknown_var:
                        copy_dir = "Unknown_True/"
                    else:
                        copy_dir = "Unknown_False/"
                
#                 orig_image = orig_image_dir + str(df_info[num_recog][0]) + "/" + df_info[num_recog][1] + ".jpg"
                orig_image = str(df_info[num_recog][0])

                valid_image = df_info[num_recog][-1]
                shutil.copy2(orig_image,self.image_save_dir + copy_dir + str(num_recog+1) + "_" + valid_image + ".jpg")
                                        
            if isRecognitionCorrect:
                self.confirmPersonIdentity(recog_results_from_file = recog_results) # save the network, analysis data, csv for learning and picture of the person in the tablet
            else:
                if isAddPersonToDB:
                    p_id = person[0]
                    num_unknown += 1
                    recog_results = []
                    if self.isMultipleRecognitions:
                        recog_values = [x for x in recogs_list if x[7] == numRecognition]
                        num_mult_recognitions = len(recog_values)
                        self.setDefinedNumMultRecognitions(num_mult_recognitions)
                        for num_recog in range(0, num_mult_recognitions):
                            recog_results.append(recog_values[num_recog][1:6])
                            if num_recog < num_mult_recognitions - 1:
                                count_recogs += 1
                    else:
                        recog_results = recogs_list[count_recogs][1:6]
                else:
                    p_id = idPerson  
                self.confirmPersonIdentity(p_id = p_id, recog_results_from_file = recog_results)

            num_recog += 1
            count_recogs += 1
            
        if self.isSaveRecogFiles:
            self.saveBN()
        
        if self.isDebugMode:
            print "time to learn:" + str(time.time() - learn_start_time) 
        return stats_openSet, stats_FR, num_recog, numNoFaceImages/(num_recog+numNoFaceImages), num_unknown

    def saveRecogniserCSV(self, recogniser_csv_file, identity_real, num_recog=None):
        """Save real identity (I), the recognition values from each identifier (F,G,A,H,T) and registration status (R), and number of the recognition (N)"""
        self.recogniser_csv_file = recogniser_csv_file
        if num_recog == 0 or num_recog is None:
            i = identity_real
            if not self.isRegistered:
                r = 1 # is registering
            else:
                r = 0 # is not registering
        else:
            i = ""
            r = ""
        num_rr = self.num_recognitions + 1
        df = pandas.DataFrame.from_items([('I', [i]), 
                                          ('F', [self.nonweighted_evidence[0]]), 
                                          ('G', [self.nonweighted_evidence[1]]),
                                          ('A', [self.nonweighted_evidence[2]]),
                                          ('H', [self.nonweighted_evidence[3]]),
                                          ('T', [self.nonweighted_evidence[4]]),
                                          ('R', [r]),
                                          ('N', [num_rr])])
        with open(recogniser_csv_file, 'a') as fd:
            df.to_csv(fd, index=False, header=False)
    
    def saveInitialRecognitionCSV(self, initial_recognition_file, recog_results, identity_est):
        """Save estimated identity (I), initial recognition values (before registration, if user hasn't enrolled yet, or equal to the values in recogniserBN.csv) (F,G,A,H,T) 
        and number of the recognition (N)"""
        self.initial_recognition_file = initial_recognition_file
        
        num_rr = self.num_recognitions + 1
        if recog_results:
            df = pandas.DataFrame.from_items([('I_est', [identity_est]), 
                                          ('F', [recog_results[0][:]]), 
                                          ('G', [recog_results[1][:]]),
                                          ('A', [recog_results[2][:]]),
                                          ('H', [recog_results[3][:]]),
                                          ('T', [recog_results[4][:]]),
                                          ('N', [num_rr])])
        else:
            df = pandas.DataFrame.from_items([('I_est', [identity_est]), 
                                          ('F', [[]]), 
                                          ('G', [[]]),
                                          ('A', [[]]),
                                          ('H', [[]]),
                                          ('T', [[]]),
                                          ('N', [num_rr])])
        with open(initial_recognition_file, 'a') as fd:
            df.to_csv(fd, index=False, header=False)
            
    def saveComparisonCSV(self, comparison_file, identity_real, identity_est, face_est, posterior_average, face_prob, calc_time, quality):
        """Save comparison file between estimated network and face recognition: 
        I_real (real identity), I_est (estimated identity by the network), F_est (estimated identity by face recognition), 
        I_prob (posterior probabilities of the network), F_prob (similarity scores from face recognition),
        Calc_time (calculation time of the network), R (registration status: 1 if registering, 0 if already enrolled),
        Quality (quality of estimation of the network), 
        Highest_I_prob (highest probability in I_prob), Highest_F_prob (highest probability in F_prob)"""
        self.comparison_file = comparison_file
        r = 0 # is not registering
        if not self.isRegistered:
            r = 1 # is registering
        if len(face_prob) > 1:
            highest_f =  max(face_prob[1:])
        else:
            highest_f = ""
        df = pandas.DataFrame.from_items([('I_real', [identity_real]), 
                                          ('I_est', [identity_est]),
                                          ('F_est', [face_est]),
                                          ('I_prob', [posterior_average]),
                                          ('F_prob', [face_prob]),
                                          ('Calc_time', [float("{0:.2f}".format(calc_time))]),
                                          ('R', [r]),
                                          ('Quality',[quality]),
                                          ('Highest_I_prob', max(posterior_average)),
                                          ('Highest_F_prob',highest_f)])
        with open(comparison_file, 'a') as fd:
            df.to_csv(fd, index=False, header=False)

    def saveDB(self, db_file, person):
        """Save database to csv or to Mongo DB: id, name, gender, birthYear, height, times, occurrence"""
        if self.isDBinCSV:
            self.saveDBToCSV(db_file, person)
        else:
            # save to mongo db
            bla = ""
#             db_handler = db.DbHandler()
#             person_dict = {"name": person[0], "gender": person[1], "birthYear": person[2], "height": person[3], "times": person[4], "occurrence": person[5]}
#             db_handler.add_person(person_dict)         
            
    def saveDBToCSV(self, db_file, person):
        """Save database to csv: id, name, gender, birthYear, height, times, occurrence"""
        df = pandas.DataFrame.from_items([('id', [person[0]]),
                                          ('name', [person[1]]), 
                                          ('gender', [person[2]]), 
#                                           ('age', [person[3]]),
                                          ('birthYear', [person[3]]),
                                          ('height', [person[4]]),
                                          ('times', [person[5]]),
                                          ('occurrence', [self.occurrences[self.i_labels.index(person[0])]])])
        self.db_df = self.db_df.append(df, ignore_index=True)      
        with open(db_file, 'a') as fd:
            df.to_csv(fd, index=False, header=False)
            
    def getAnalysisData(self, recog_results, identity_real, ie):
        """Get results of all the parameters in the system for analysis: 
        Database (identity labels in the database), image_id, I_real (real identity), I_est (estimated identity), 
        for all parameters I, F, G, A, H, T: cpt (likelihood in the network before update), posterior (inference posterior result), recognition results"""
        i_post = ie.posterior(self.I)[:]
        i_max_cpt = np.max(ie.posterior(self.I)[:])
        identity_est = self.i_labels[np.argmax(ie.posterior(self.I)[:])]
        isclose_ar = np.isclose(i_post, i_max_cpt)
        if np.isclose(i_post, i_max_cpt).all():
            # if all states are equally likely then the person is unknown
            identity_est = "unknown-equal"
        elif len(isclose_ar[isclose_ar==True]) > 1:
            # if maximum appears more than one time in the array
            identity_est = "unknown-max-equal"

        date_today = recog_results[4][2] + " " + recog_results[4][3] + " " + recog_results[4][4] + " " + recog_results[4][0]
        date_now = str(datetime.strptime(date_today, '%d %B %Y %H:%M:%S'))          
        if self.image_id is None:
            self.image_id = identity_real + "_" + "0001"
        data = OrderedDict([("Date", date_now),
                ("Database", self.i_labels),
                ("Image_id", self.image_id),
                ("I_real", identity_real),
                ("I_est", [identity_est, i_max_cpt]),
                ("I_cpt", self.r_bn.cpt(self.I)[:].tolist()),
                ("I_posterior", ie.posterior(self.I)[:].tolist()),
                ("F_est", recog_results[0]),
                ("F_cpt", self.r_bn.cpt(self.F)[:].tolist()),
                ("F_posterior", ie.posterior(self.F)[:].tolist()),
                ("G_est", recog_results[1]),
                ("G_cpt", self.r_bn.cpt(self.G)[:].tolist()),
                ("G_posterior", ie.posterior(self.G)[:].tolist()),
                ("A_est", recog_results[2]),
                ("A_cpt", self.r_bn.cpt(self.A)[:].tolist()),
                ("A_posterior", ie.posterior(self.A)[:].tolist()),
                ("H_est", recog_results[3]),
                ("H_cpt", self.r_bn.cpt(self.H)[:].tolist()),
                ("H_posterior", ie.posterior(self.H)[:].tolist()),
                ("T_est", recog_results[4]),
                ("T_cpt", self.r_bn.cpt(self.T)[:].tolist()),
                ("T_posterior", ie.posterior(self.T)[:].tolist())])
        
        return data
    
    def saveAnalysisFile(self, recog_results, identity_real, ie, isPrevSavedToAnalysis, num_recog = None):
        if self.isDBinCSV:
            if num_recog:
                self.saveAnalysisToJson(recog_results, identity_real, ie, isPrevSavedToAnalysis, num_recog = num_recog)
            else:
                self.saveAnalysisToJson(recog_results, identity_real, ie, isPrevSavedToAnalysis)
    
    def saveAnalysisToJson(self, recog_results, identity_real, ie, isPrevSavedToAnalysis, num_recog = None):
        """Save analysis to json file"""
                
        a = []
        if self.isMultipleRecognitions and num_recog is not None and num_recog < self.num_mult_recognitions - 1:
            self.analysis_data_list.append(self.getAnalysisData(recog_results, identity_real, ie))
        else:
            if self.isMultipleRecognitions:
                self.analysis_data_list.append(self.getAnalysisData(recog_results, identity_real, ie))
                a = self.analysis_data_list
                num_file = self.num_recognitions + 1

            else:
                dt = self.getAnalysisData(recog_results, identity_real, ie)
                a.append(dt)
                num_file = self.num_recognitions + 1
                
            if isPrevSavedToAnalysis:
                fname = self.analysis_file.replace(".json","") + str(num_file) + "_2.json"
            else:
                fname = self.analysis_file.replace(".json","") + str(num_file) + ".json"
            with open(fname, mode='w') as f:
                f.write(json.dumps(a, ensure_ascii=False, indent=2))

    
    def saveConfusionMatrix(self, comparison_file=None, conf_matrix_file=None):
        """Save confusion matrix to file for both face recognition and network results"""
        if comparison_file is None:
            comparison_file = self.comparison_file
        if conf_matrix_file is None:
            conf_matrix_file = self.conf_matrix_file
                        
        df_comp = pandas.read_csv(comparison_file, usecols =["I_real", "I_est", "F_est", "R"])
        comp_list = df_comp.values.tolist()
        conf_matrices = [[[0 for _ in range(0, self.num_people+1)] for i in range(0, self.num_people)] for j in range(0,2)]
        for num_recog in range(0, len(comp_list)):
            i_real = comp_list[num_recog][0]
            for num_estimator in range(0, 2): # I and F
                identity_est = comp_list[num_recog][num_estimator+1]
                if comp_list[num_recog][-1] == 1: # unknown
                    conf_matrices[num_estimator][0][identity_est] += 1
                    conf_matrices[num_estimator][0][-1] += 1
                    if identity_est != int(self.unknown_var):
                        conf_matrices[num_estimator][i_real][identity_est] += 1
                        conf_matrices[num_estimator][i_real][-1] += 1          
                else:
                    conf_matrices[num_estimator][i_real][identity_est] += 1
                    conf_matrices[num_estimator][i_real][-1] += 1
        percent_conf_matrices = [[[0 for _ in range(0, self.num_people+1)] for i in range(0, self.num_people)] for j in range(0,2)]
        for num_estimator in range(0, 2): # I and F
            conf_matrix = conf_matrices[num_estimator][:]
            row_counter = 0
            for row in conf_matrix:
                num_recog_person = row[-1]
                if num_recog_person > 0:
                    percent_row = [float("{0:.2f}".format(row[i]*1.0/num_recog_person)) for i in range(0, len(row)-1)]
                    percent_row.append(num_recog_person)
                    percent_conf_matrices[num_estimator][row_counter] = percent_row
                row_counter += 1
                
        file_names = ["Network", "FaceRecognition"]
        
        for num_estimator in range(0,2): # I and F
            file_name = file_names[num_estimator]
            conf_matrix_fi = conf_matrix_file.replace(".csv", file_name + ".csv")
            conf_matrix = conf_matrices[num_estimator][:]
            with open(conf_matrix_fi, 'wb') as outcsv:
                writer = csv.writer(outcsv)
                row = [str(i) for i in range(0, self.num_people)]
                row.insert(0,"Real/Estimate")
                row.append("Num_recog")
                writer.writerow(row)
                
            with open(conf_matrix_fi, 'a') as outcsv:
                writer = csv.writer(outcsv)
                row_counter = 0
                for row in conf_matrix:
                    row.insert(0,row_counter)
                    writer.writerow(row)
                    row_counter += 1

        for num_estimator in range(0,2): # I and F
            file_name = file_names[num_estimator]
            conf_matrix_fi = conf_matrix_file.replace(".csv", file_name + "Percent.csv")
            percent_conf_matrix = percent_conf_matrices[num_estimator][:]
            with open(conf_matrix_fi, 'wb') as outcsv:
                writer = csv.writer(outcsv)
                row = [str(i) for i in range(0, self.num_people)]
                row.insert(0,"Real/Estimate")
                row.append("Num_recog")
                writer.writerow(row)
                
            with open(conf_matrix_fi, 'a') as outcsv:
                writer = csv.writer(outcsv)
                row_counter = 0
                for row in percent_conf_matrix:
                    row.insert(0,row_counter)
                    writer.writerow(row)
                    row_counter += 1
                
    def copyNetworkDBFromValidation(self, val_folder, test_folder):
        """Copy network (bif file) and db (db.csv) from training folder to test folder"""
        shutil.copy2(val_folder+self.recog_file, test_folder)
        df_db = pandas.read_csv(val_folder+self.db_file, dtype={"id": object}) 
        db_list = df_db.values.tolist()
        with open(test_folder+self.db_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(df_db.columns.values) 

        for person in db_list:
            person[-1] = [1,1,1]
            with open(test_folder+self.db_file, 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(person)

    #---------------------------------------------STATS FUNCTIONS---------------------------------------------#

            
    def getPerformanceMetrics(self, identity_est, real_identity, unknown_var, isRegistered, stats_openSet):
        """Returns [#correct_known_recognitions (if the user is known and the recognition is correct), #incorrect_unknown_recognitions (if the user is unknown, but estimated as known) ]"""
        if isRegistered and identity_est == real_identity:
            stats_openSet[0] += 1
        elif not isRegistered and identity_est != unknown_var:
            stats_openSet[1] += 1
        return stats_openSet

    def getDIRFAR(self, stats_openSet, num_recog, num_unknown):
        """Get detection and identification rate (DIR) = #correct_known_recognitions/#known_recognitions
        False alarm rate (FAR) = # incorrect_unknown_recognitions/# unknown_recognitions. 
        Returns [DIR, FAR]"""
        stats_openSet[0] = float("{0:.3f}".format(stats_openSet[0]/((num_recog-num_unknown)*1.0)))
        if num_unknown > 0:
            stats_openSet[1] = float("{0:.3f}".format(stats_openSet[1]/(num_unknown*1.0)))
        else:
            stats_openSet[1] = 0.0
                  
        return stats_openSet
        
    def getDetailedRecogRates(self, i_real_list, i_est_list, f_est_list, r_list, num_unknown, num_recog):
        """
        Get detailed recognition rates [true_positive, true_negative, [false_positive_another_case, false_positive_unknown_case], false_negative] for network and face recognition respectively
        Returns stats (real numbers), stats_percent (percentage of the values), stats_graph ((TP+TN)/num_recog, DIR, 1-FAR )
        """
        stats = [[0,0,[0,0],0],[0,0,[0,0],0]] # [true_positive, true_negative, [false_positive_another_case, false_positive_unknown_case], false_negative] for network and face recognition respectively

        stats_percent = [[0,0,[0,0],0],[0,0,[0,0],0]]
        
        stats_graph = [[0,0,0],[0,0,0]]
        
        for recog_type in range(0, len(stats)):
            for count in range(0, num_recog):
                if recog_type == 0:
                    i_est = i_est_list[count]
                elif recog_type == 1:
                    i_est = f_est_list[count]
                
                if i_real_list[count] == i_est:
                    stats[recog_type][0] += 1
                elif i_est == self.unknown_var and r_list[count]:
                    stats[recog_type][1] += 1
                elif i_est == self.unknown_var and not r_list[count]:
                    stats[recog_type][3] += 1
                elif i_est != self.unknown_var and r_list[count]:
                    stats[recog_type][2][1] += 1
                else:
                    stats[recog_type][2][0] += 1
                    
        for stats_counter in range(0, len(stats)):
            for ss_counter in range(0, len(stats[stats_counter])):
                if ss_counter != 2:
                    stats_percent[stats_counter][ss_counter] = float("{0:.3f}".format(stats[stats_counter][ss_counter]/(num_recog*1.0)))
                else:
                    stats_percent[stats_counter][ss_counter] = [float("{0:.3f}".format(stats[stats_counter][ss_counter][0]/(num_recog*1.0))), float("{0:.3f}".format(stats[stats_counter][ss_counter][1]/(num_recog*1.0)))]
        
        for stats_counter in range(0, len(stats)):
            for ss_counter in range(0, 3):
                if ss_counter == 0:
                    stats_graph[stats_counter][ss_counter] = (stats[stats_counter][0] + stats[stats_counter][1])/(num_recog*1.0)
                elif ss_counter == 1:
                    if num_recog > num_unknown:
                        stats_graph[stats_counter][ss_counter] = stats[stats_counter][0]/((num_recog-num_unknown)*1.0)
                    else:
                        stats_graph[stats_counter][ss_counter] = 0
                elif ss_counter == 2:
                    if num_unknown > 0:                
                        stats_graph[stats_counter][ss_counter] = stats[stats_counter][1]/(num_unknown*1.0)
                    else:
                        stats_graph[stats_counter][ss_counter] = 0
                    
                    
        return stats, stats_percent, stats_graph
        
    def getHeightStddev(self, recogniser_csv_file):
        """
        Get standard deviation of height from recognition file. 
        Returns the list of standard deviation from true height (ground truth), standard deviation within estimates for each user
        """
        df = pandas.read_csv(recogniser_csv_file, dtype={"I": object}, usecols =["I", "H"], converters={"H": ast.literal_eval})
        group_v = df.loc[:,['I','H']].groupby('I')
        std_dev = [0.0 for i in range(1, len(self.i_labels))]
        std_dev_est = [0.0 for i in range(1, len(self.i_labels))]
        for counter in range(1,len(self.i_labels)):
            true_height = float(self.heights[counter])
            gr = group_v.get_group(self.i_labels[counter])
            avg_val = 0
            for g_counter in range(0, len(gr)):
                l_val = gr.iloc[g_counter,1]
                est = l_val[0]
                std_dev[counter-1] += math.pow(est - true_height, 2)
                avg_val += est
 
            if len(gr) > 0:
                std_dev[counter-1] = math.sqrt(std_dev[counter-1]/len(gr))            
                avg_val /= len(gr)
                 
            for g_counter in range(0, len(gr)):
                l_val = gr.iloc[g_counter,1]
                est = l_val[0]
                std_dev_est[counter-1] += math.pow(est - avg_val, 2)
                 
            if len(gr) > 1:
                std_dev_est[counter-1] = math.sqrt(std_dev_est[counter-1]/(len(gr)-1))
            
        return std_dev, std_dev_est
    
    def getAgeStddev(self, recogniser_csv_file, initial_recognition_file):
        """
        Get standard deviation of age from recognition file. 
        Returns the list of standard deviation from true age (ground truth), standard deviation within estimates for each user
        """
        df_final = pandas.read_csv(recogniser_csv_file, dtype={"I": object}, usecols =["I", "A", "R", "N"], converters={"A": ast.literal_eval})
        df_init = pandas.read_csv(initial_recognition_file, usecols =["I_est", "A", "N"], converters={"A": ast.literal_eval})
        
        recogs_list = df_final.values.tolist()
        count_recogs = 0
        stddev_true_mean = [0.0 for i in range(1, len(self.i_labels))]
        stddev_est_list = [0.0 for i in range(1, len(self.i_labels))]    
        avg_val = [0.0 for i in range(1, len(self.i_labels))]
        estimates_mean = [[] for i in range(1, len(self.i_labels))]
        estimates_stddev = [[] for i in range(1, len(self.i_labels))]
        while count_recogs < len(recogs_list):
            isRegistered =  not recogs_list[count_recogs][2]# False if register button is pressed (i.e. if the person starts the session for the first time)
            numRecognition = recogs_list[count_recogs][3]
            p_id = recogs_list[count_recogs][0]
            p_id_index = self.i_labels.index(p_id)

            if isRegistered:
                if self.isMultipleRecognitions:
                    num_mult_recognitions = df_final.loc[df_final['N'] == numRecognition].A.count()

                    for num_recog in range(0, num_mult_recognitions):             
                        est_mean = recogs_list[count_recogs][1][0]
                        est_conf = recogs_list[count_recogs][1][1]
                        if est_conf > 0: 
                            if est_conf == 1.0:
                                est_stddev = 0.0
                            else:
                                est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                            estimates_mean[p_id_index-1].append(est_mean)
                            estimates_stddev[p_id_index-1].append(est_stddev)
                            stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                            avg_val[p_id_index-1] += est_mean
                        if num_recog < num_mult_recognitions - 1:
                            count_recogs += 1
                else:
                    est_mean = recogs_list[count_recogs][1][0]
                    est_conf = recogs_list[count_recogs][1][1]
                    if est_conf > 0:  
                        if est_conf == 1.0:
                            est_stddev = 0.0
                        else:
                            est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                        estimates_mean[p_id_index-1].append(est_mean)
                        estimates_stddev[p_id_index-1].append(est_stddev)
                        stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                        avg_val[p_id_index-1] += est_mean
                    
            else:
                if self.isMultipleRecognitions:
                    
                    init_recog_est = df_init.loc[df_init['N'] == numRecognition].values.tolist()
                    num_mult_recognitions = len(init_recog_est)
                    for num_recog in range(0, num_mult_recognitions):
                        est_mean = init_recog_est[num_recog][1][0]
                        est_conf = init_recog_est[num_recog][1][1]
                        if est_conf > 0:  
                            if est_conf == 1.0:
                                est_stddev = 0.0
                            else:
                                est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                            estimates_mean[p_id_index-1].append(est_mean)
                            estimates_stddev[p_id_index-1].append(est_stddev)
                            stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                            avg_val[p_id_index-1] += est_mean
                        
                    num_mult_recognitions = df_final.loc[df_final['N'] == numRecognition].A.count()
                    for num_recog in range(0, num_mult_recognitions):
                        est_mean = recogs_list[count_recogs][1][0]
                        est_conf = recogs_list[count_recogs][1][1]
                        if est_conf > 0:  
                            if est_conf == 1.0:
                                est_stddev = 0.0
                            else:
                                est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                            estimates_mean[p_id_index-1].append(est_mean)
                            estimates_stddev[p_id_index-1].append(est_stddev)
                            stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                            avg_val[p_id_index-1] += est_mean
                        if num_recog < num_mult_recognitions - 1:
                            count_recogs += 1

                else:

                    init_recog_est = df_init.loc[df_init['N'] == numRecognition].values.tolist()
                    est_mean = init_recog_est[1][0]
                    est_conf = init_recog_est[1][1]
                    if est_conf > 0:  
                        if est_conf == 1.0:
                            est_stddev = 0.0
                        else:
                            est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                        estimates_mean[p_id_index-1].append(est_mean)
                        estimates_stddev[p_id_index-1].append(est_stddev)
                        stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                        avg_val[p_id_index-1] += est_mean
                    
                    est_mean = recogs_list[count_recogs][1][0]
                    est_conf = recogs_list[count_recogs][1][1]
                    if est_conf > 0:  
                        if est_conf == 1.0:
                            est_stddev = 0.0
                        else:
                            est_stddev = 0.5/self.normppf(est_conf + (1-est_conf)/2.0)
                        estimates_stddev[p_id_index-1].append(est_stddev)
                        stddev_true_mean[p_id_index-1] += math.pow(est_mean - self.ages[p_id_index], 2) + math.pow(est_stddev,2)
                        avg_val[p_id_index-1] += est_mean
            count_recogs += 1
        
        for counter in range(0, len(estimates_mean)):
            if len(estimates_mean[counter]) > 0:
                avg_val[counter] /= len(estimates_mean[counter])
                stddev_true_mean[counter] = math.sqrt(stddev_true_mean[counter]/len(estimates_mean[counter]))  
            for count_val in range(0, len(estimates_mean[counter])):
                stddev_est_list[counter] += math.pow(estimates_mean[counter][count_val] - avg_val[counter], 2) + math.pow(estimates_stddev[counter][count_val],2)
            if len(estimates_mean[counter]) > 1:
                stddev_est_list[counter] = math.sqrt(stddev_est_list[counter]/(len(estimates_mean[counter])-1))
                
        return stddev_true_mean, stddev_est_list
        
            
    def getTimeStddev(self, recogniser_csv_file, recog_folder):
        """
        Get standard deviation of time from recognition file.
        Returns the list of standard deviation within time of interactions for each user 
        """
        df = pandas.read_csv(recogniser_csv_file, dtype={"I": object}, usecols =["I", "T"], converters={"T": ast.literal_eval})
        group_v = df.loc[:,['I','T']].groupby('I')
        std_dev_est = [0.0 for i in range(1, len(self.i_labels))]
        values = []
        for counter in range(1,len(self.i_labels)):
            t_values = []
            gr = group_v.get_group(self.i_labels[counter])
            avg_val = 0
            for g_counter in range(0, len(gr)):
                l_val = gr.iloc[g_counter,1]
                est = self.getTimeSlot(l_val)
                t_values.append(est)
                avg_val += est
            
            values.append(t_values)
            if len(gr) > 0:
                avg_val /= len(gr)
                 
            for g_counter in range(0, len(gr)):
                l_val = gr.iloc[g_counter,1]
                est = self.getTimeSlot(l_val)
                std_dev_est[counter-1] += math.pow(est - avg_val, 2)
                 
            if len(gr) > 1:
                std_dev_est[counter-1] = math.sqrt(std_dev_est[counter-1]/(len(gr)-1))
        
        times_curves = []
        for v in values:
            time_curve = []
            for v_counter in range(0, len(v)):
                t_curve = self.getCurve(mean = v[v_counter], stddev = self.stddev_time, min_value = self.time_min, max_value = self.time_max, weight = 1.0)
                if v_counter == 0:
                    time_curve = t_curve[:]
                else:
                    time_curve = [x + y for x, y in zip(time_curve, t_curve)]
            time_curve = self.normaliseSum(time_curve)
            times_curves.append(time_curve)
        
        with open(recog_folder + "time_stddev.csv", 'wb') as outcsv:
            writer = csv.writer(outcsv)
            for row_counter in range(0, len(times_curves[0])):
                row = [row_counter]
                for time_curve_counter in range(0, len(times_curves)):
                    row.append(times_curves[time_curve_counter][row_counter])
                writer.writerow(row)
                
        return std_dev_est
     

    def getRangeVarDetectionRate(self, var_est, var_real, bin_size, stats_var, decimals=0):
        """
        Get range variable detection rate: if estimated_value/ bin_size = real_value -> correct recognition, else false
        """
        if np.around(var_est, decimals=decimals)/bin_size == var_real:
           stats_var += 1
        return stats_var
    
    def getLabelVarDetectionRate(self, var_est, var_real, stats_var):
        """
        Get label variable detection rate: if estimated_value = real_value -> correct recognition, else false
        """
        if var_est == var_real:
            stats_var += 1
        return stats_var
      
    def getParamStats(self, idPerson, recog_results, clean_times, stats_gender, stats_age, stats_height, stats_time, isAddPersonToDB, person = []):
        """Returns the stats (number of correct recognitions) of gender, age, height, and time for the specified person (or id of person)"""
        person_ind = int(idPerson)
        
        if isAddPersonToDB:
            real_gender = person[2]
            real_age = person[3]
            real_height = person[4]
        else:
            real_gender = self.genders[person_ind]
            real_age = self.ages[person_ind]
            real_height = self.heights[person_ind]
            
        stats_gender = self.getLabelVarDetectionRate(recog_results[1][0], real_gender, stats_gender)
        stats_age = self.getRangeVarDetectionRate(recog_results[2][0], real_age, self.age_bin_size, stats_age)
        stats_height = self.getRangeVarDetectionRate(recog_results[3][0], real_height, self.height_bin_size, stats_height)
        stats_time = self.getTimeDetectionRate(recog_results[4], clean_times[person_ind-1], stats_time)
        return stats_gender, stats_age, stats_height, stats_time

    #---------------------------------------------FUNCTIONS FOR IMAGES---------------------------------------------# 
    
    def saveImageAfterRecognition(self, isRegistered, p_id):
        """Saves image(s) after recognition: by calling saveImageToTablet method. 
        If recognition is offline, only fills the image_id value (for analysis purposes)"""
        p_id = str(p_id)
        self.p_id = p_id
        self.image_id = None
        if self.recog_results_from_file is None:
            if self.isMultipleRecognitions:
                num_saves = 0
                for num_recog in range(0, self.def_num_mult_recognitions): # save images even if a face is not detected
                    if num_recog in self.discarded_data:
                        self.image_id = self.saveImageToTablet(p_id, num_recog = num_recog, num_saves = num_saves, isDiscardedImage=True)
                    else:
                        self.image_id = self.saveImageToTablet(p_id, num_recog = num_recog, num_saves = num_saves, identity_est = self.identity_est_list[num_recog], isRegistered = isRegistered)
                    num_saves += 1
                
            else:
                self.image_id = self.saveImageToTablet(p_id, identity_est = self.identity_est, isRegistered = isRegistered)
        else:
            if self.image_id is None:
                if isRegistered: 
                    num_matches = self.occurrences[self.i_labels.index(p_id)][0]
                    orig_matches = num_matches
                    counter = 0
                    for i in range(0,4):
                        if num_matches/10 != 0:
                            num_matches = num_matches/10
                            counter += 1 
                        else:
                            counter += 1 
                            break
                    self.image_id = str(self.num_recognitions+1) + "_" + p_id + "_" + str(orig_matches)
                else:
                    self.image_id =  str(self.num_recognitions+1) + "_" + p_id + "_" + "1"      

    def saveImageToTablet(self, p_id, num_recog=None, num_saves=None, isDiscardedImage=False, imageName=None, identity_est = None, isRegistered= False):
        """
        Saves images from the specified directory (self.imagePath or self.imageToCopy) to a folder specifying the correctness of the recognition: 
        Known_True/, Known_Unknown/, Known_False/, Unknown_True/, Unknown_False/ 
        Name of image:
        single recognition: str(num_recognitions + 1) + "_" + p_id + "_" + str(orig_matches) + ".jpg"
        multiple recognitions: str(self.num_recognitions + 1) + "_" + p_id + "_" + str(orig_matches) + "-" + str(num_saves) +".jpg" 
        """
        # TODO: check with windows (/ might need to be \ instead)
        
        if self.isMemoryOnRobot:
            image_dir = self.image_save_dir

            if self.imageToCopy is not None:
                temp_image = self.imageToCopy
            else:
                temp_image = self.imagePath
        else:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            temp_dir = os.path.abspath(os.path.join(cur_dir, '../..', 'cam')) + "/"
            image_dir = os.path.abspath(os.path.join(cur_dir, '', 'images')) + "/"

            if self.imageToCopy is not None:
                temp_image = self.imageToCopy
            else:
                temp_image = temp_dir + "temp.jpg"
                
        if isDiscardedImage:
            image_dir += "discarded/"
        elif identity_est is not None:
            if isRegistered:
                if identity_est == p_id:
                    image_dir += "Known_True/"
                elif identity_est == self.unknown_var:
                    image_dir += "Known_Unknown/"
                else:
                    image_dir += "Known_False/"
            else:
                if identity_est == self.unknown_var:
                    image_dir += "Unknown_True/"
                else:
                    image_dir += "Unknown_False/"

        if p_id in self.i_labels:
            num_matches = self.occurrences[self.i_labels.index(p_id)][0] + 1
        else:
            num_matches = 1
        orig_matches = num_matches
        counter = 0 
        for i in range(0,4):
            if num_matches/10 != 0:
                num_matches = num_matches/10
                counter += 1 
            else:
                counter += 1 
                break

        if self.isMultipleRecognitions:

            if imageName is not None:
                to_rep = image_dir + imageName + "-" + str(num_recog) + ".jpg"
            else:
                to_rep = str(num_recog) + ".jpg"
            temp_image = self.imagePath.replace(".jpg", to_rep)

            save_name = image_dir + str(self.num_recognitions + 1) + "_" + p_id + "_" + str(orig_matches) + "-" + str(num_saves) +".jpg" 

        else:
            if imageName is not None:
                save_name = image_dir + imageName + ".jpg"  
            else: 
                save_name = image_dir + str(self.num_recognitions + 1) + "_" + p_id + "_" + str(orig_matches) + ".jpg"
        
        shutil.copy2(temp_image,save_name)

        image_id = str(self.num_recognitions + 1) + "_" + p_id + "_" + str(orig_matches)
        return image_id
            
    #---------------------------------------------FUNCTIONS FOR THE ROBOT---------------------------------------------#

    def connectToRobot(self, ip, port=9559, useSpanish = False, isImageFromTablet = True, isMemoryOnRobot = False, imagePath = ""):
        """Connect to the robot and set robot parameters of the recognition (NAOqi)"""
        self.robot_ip = ip
        self.robot_port = port
        self.session = qi.Session()
        try:
            self.session.connect("tcp://" + ip + ":" + str(port))
        except RuntimeError:
            logging.debug("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)
        self.animatedSpeechProxy = self.session.service("ALAnimatedSpeech")
        self.tts = self.session.service("ALTextToSpeech")
        self.configuration = {"bodyLanguageMode":"contextual"}
        self.useSpanish = useSpanish
        self.face_service = self.session.service("ALFaceDetection")
        self.people_service = self.session.service("ALPeoplePerception")
        self.memory_service = self.session.service("ALMemory")
        self.recog_service = self.session.service("RecognitionService")
        self.isImageFromTablet = isImageFromTablet
        
        self.isMemoryOnRobot = isMemoryOnRobot
        if self.isMemoryOnRobot:
            self.isDBinCSV = True
        self.imagePath = imagePath
        
        self.recog_service.initSystem(self.useSpanish, self.isImageFromTablet, imagePath) # initialize the robot breathing, height offset, and language

    def recognisePerson(self, num_recog = None):
        """Recognise person using NAOqi. Modify this function if another identifiers are used for modalities.
        Returns recog_results in the form: 
        [[0.64, [['2', 0.8], ['1', 0.3]]], ['Female', 0.6], [28L, 0.5], [165.0, 0.0], ['18:31:14', '3', '06', 'June', '2017']] 
        i.e. 
        [
        [accuracy, [['ID_1', sim_score_1], ['ID_2', sim_score_2],..], 
        ['Female', conf_score_gender], 
        [age, conf_score_age], 
        [height, conf_score_height], 
        ['HH:MM:SS', 'num_day', 'Day', 'Month_name', 'Year'] (num_day is the number of day in the week, Monday is 1, Tuesday 2, etc. Day, month and year are not necessary
        ]
        
        """
        if self.recog_results_from_file is None: 
            if self.isMultipleRecognitions:
                self.recog_service.setImagePathMult(num_recog)
            if self.isMemoryOnRobot:
                recog_results = self.recog_service.recognisePerson()
            else:
                self.recog_service.subscribeToPeopleDetected()
                self.event_recog = threading.Event()
                self.subscribeToRecognitionResultsUpdated()
                self.event_recog.wait()
                recog_results = self.recog_temp
        else:
            if self.isMultipleRecognitions:
                recog_results = self.recog_results_from_file[num_recog]
            else:
                recog_results = self.recog_results_from_file
        if self.isDebugMode:
            print "recog_results: " + str(recog_results)
        return recog_results
            
    def subscribeToRecognitionResultsUpdated(self):
        """Subscribe to recognition results updated (NAOqi). 
        'RecognitionResultsUpdated' signal is given by the NAOqi when recognition results are filled for the current image, hence, network can estimate the identity"""
        self.recognitionResultsUpdatedEvent = "RecognitionResultsUpdated"
        self.recognitionResultsUpdated = self.memory_service.subscriber(self.recognitionResultsUpdatedEvent)
        self.idRecognitionResultsUpdated = self.recognitionResultsUpdated.signal.connect(functools.partial(self.onRecognitionResultsUpdated, self.recognitionResultsUpdatedEvent))
        self.recog_service.subscribeToPeopleDetected()
      
    def onRecognitionResultsUpdated(self, strVarName, value):
        """When the RecognitionResultsUpdated signal is received, get the recognition results (NAOqi)."""
        self.recognitionResultsUpdated.signal.disconnect(self.idRecognitionResultsUpdated)
        self.recognitionResultsUpdated = None
        self.idRecognitionResultsUpdated = -1
        self.recog_temp = value
        self.event_recog.set()
    
    def threadedLearnPerson(self, num_recog):
        """Parallel learning of images (NAOqi)"""
        self.recog_service.setImagePathMult(num_recog)
        learn_face_success = self.recog_service.addPictureToPerson(self.p_id)

        return learn_face_success
        
    def learnPerson(self, isRegistered, p_id, isRobotLearning):
        """Learn image for the face recognition dataset (NAOqi)"""
        self.p_id = p_id
        self.image_id = None
        if self.recog_results_from_file is None and isRobotLearning:
            if isRegistered:     
                if self.isMultipleRecognitions:
                    # Parallel learning takes longer than sequential learning
    #                 pool = ThreadPool(self.num_mult_recognitions)
    #                 joint_results = pool.map(self.threadedLearnPerson, [i for i in range(0, self.num_mult_recognitions)])
    #                 pool.close()
    #                 pool.join()
    #                 print "time to learn parallel: " + str(time.time() - p_start_time)
                    for num_recog in [item for item in range(0, self.def_num_mult_recognitions) if item not in self.discarded_data]:
                        self.recog_service.setImagePathMult(num_recog)
                        learn_face_success = self.recog_service.addPictureToPerson(p_id)
                    
                else:
                    learn_face_success = self.recog_service.addPictureToPerson(p_id)
            else:
                if not self.isMemoryOnRobot:
                    # registerPersonOnRobot is called in recognitionModule if self.isMemoryOnRobot
                    if self.isMultipleRecognitions:
                        for num_recog in [item for item in range(0, self.def_num_mult_recognitions) if item not in self.discarded_data]:
                            self.recog_service.setImagePathMult(num_recog)
                            learn_face_success = self.recog_service.registerPerson(p_id)
                            if num_recog == 0:
                                counter = 1
                                while not learn_face_success and counter < 3:
                                    # TODO: take picture here (an example in recognitionModule for this for NaoQi)
                                    learn_face_success = self.recog_service.registerPerson(p_id) 
                    else:
                        learn_face_success = self.recog_service.registerPerson(p_id)
                        counter = 1
                        while not learn_face_success and counter < 3:
                            # TODO: take another picture from tablet and send to robot (an example in recognitionModule for this for NaoQi)
                            learn_face_success = self.recog_service.registerPerson(p_id)
                elif isRobotLearning:
                    if self.isMultipleRecognitions:
                        for num_recog in [item for item in range(0, self.def_num_mult_recognitions) if item not in self.discarded_data]:
                            self.recog_service.setImagePathMult(num_recog)
                            learn_face_success = self.recog_service.registerPerson(p_id)
                    else:
                        learn_face_success = self.recog_service.registerPerson(p_id)

 
    def resetFaceDetectionDB(self):
        """Reset face detection database (NAOqi)"""
        self.face_service.clearDatabase()

    def saveFaceDetectionDB(self):
        """Save face recognition database to the recognition folder (NAOqi)"""
        db_path = self.face_service.getUsedDatabase()
        recog_face_path = os.path.dirname(os.path.realpath(__file__)) + "/" + self.faceDB
        if db_path != recog_face_path:
            # if the robot is using another face directory other than the current directory, save the file to the recognition folder.
            shutil.copy2(db_path, self.recog_folder)
    
    def useFaceDetectionDB(self, facedb=None):
        """Use the specified facedb ('faceDB') in the current directory and recog_folder (NAOqi)"""
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        if facedb:
            self.faceDB = self.recog_folder + facedb
        self.face_service.useDatabase(cur_dir + "/" + self.faceDB)
    
    def removePersonFromFaceDB(self, person_id):
        self.face_service.forgetPerson(person_id)
        
    def testImagesForFace(self, src_folder):
        """Test images for face detection. Remove images that do not have a face detected (NAOqi)"""
        subdirs = [x[0] for x in os.walk(src_folder)]
        for subdir in subdirs[1:]:
            # print subdir
            files = os.walk(subdir).next()[2]
            for file in files:
                # print file
                for i in range(0,2):
                    self.recog_service.setImagePath(subdir + "/" + file)
                    self.recog_service.setHeightOfPerson(165, True)
                    self.recog_service.setTimeOfPerson(['21:36:10', '7'])
                    self.recog_results_from_file = None
                    self.isMemoryOnRobot = True
                    self.isMultipleRecognitions = False
                    recog_results = self.recognisePerson()
                    # print recog_results
                    if recog_results == []:
                        os.remove(subdir + "/" + file)
                        # print "deleted"
                        break
                time.sleep(0.5)        


    def loadSentencesForRecognition(self):
        """Load sentences for recognition feedbacks."""
        self.lookAtTablet = "Hello there, could you look at the tablet "
        self.pleasePhrase = "please?"
        self.enterName = "and enter your name please?"
        self.unknownPerson = "Oh I'm sorry, I couldn't recognise who you are! Could you enter your name on the tablet please?"
        self.askForIdentityConfirmal = "Hello XX, it is nice to see you again! Could you confirm that it is you please?"   
        self.falseRecognition = ["Ah, of course, my apologies! My eyes seem to fail me.. Welcome back XX!", "You look different today XX, is it a new haircut?"]
        self.registrationPhrase = "Hello XX, nice to meet you!"
        self.falseRegistration = "But we have met before! Nice to see you again XX!"
        self.correctRecognition = ["I knew it was you, just wanted to be sure!", "You look very good today XX!"]
        self.noFaceInImage = "I am sorry, there seems to be a problem with the image. Could you look at the tablet again please?"
        if self.isMemoryOnRobot:
            self.registrationPhrase = "Nice to meet you XX!"
            self.falseRecognition = ["Ah, of course, my apologies! My eyes seem to fail me.. Nice to see you again XX!", "You look different today XX, is it a new haircut?", 
                                         "Ehehe, I was kidding. Of course it is you XX!", "Well we robots can be wrong sometimes, but we have you, XX, to make us better."]
            self.unknownPerson = "Hello there, I am Pepper! What is your name?"
            self.correctRecognition = ["I knew it was you, just wanted to be sure!", "You look very good today XX!", "Just wanted to say hello, hope you are doing fine XX!", 
                                           "I feel much better every time I see you XX!", "You bring much needed sunshine to my day XX!"]
            
    def say(self, sentence):
        """Say the specified sentence (NAOqi)"""
        self.tts.setVolume(0.85)
        self.tts.setParameter("speed", 95)
        threading.Thread(target = self.animatedSpeechProxy.say, args=(sentence,self.configuration)).start()

    #---------------------------------------------FUNCTIONS FOR CROSS VALIDATION---------------------------------------------#
    
    
    def runCrossValidation(self, num_people, training_folder, test_folder, db_list=None, init_list=None, recogs_list=None, bn=None, isTestData = False,
                               weights = None, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = "evidence", probThreshold = None,
                               isMultRecognitions = False, num_mult_recognitions = None, qualityCoefficient = None,
                               db_file = None, init_recog_file = None, final_recog_file = None, valid_info_file = None, isSaveRecogFiles = True, isSaveImageAn = True):
        
        """Run cross validation offline from recognition files or recognition lists"""
        
        start_time_run = time.time()
        
        """BEGIN: set params"""
        
        self.isSaveRecogFiles = isSaveRecogFiles
        
        recog_folder = ""
        
        if isTestData:
            recog_folder = test_folder
            if self.isSaveRecogFiles:
                self.resetFilePaths()
                self.setFilePaths(recog_folder)
                self.resetFiles()
                self.resetFilePaths()
                self.copyNetworkDBFromValidation(training_folder, test_folder)
                self.setFilePaths(recog_folder)
            self.setUpdateMethod("none")
        else:
            recog_folder = training_folder
            if self.isSaveRecogFiles:
                self.resetFilePaths()
                self.setFilePaths(recog_folder)
                self.resetFiles()
            self.setUpdateMethod(updateMethod)
        
        if weights is not None:
            self.setWeights(weights[0], weights[1], weights[2], weights[3], weights[4])
            
        if faceRecogThreshold is not None:
            self.setFaceRecognitionThreshold(faceRecogThreshold)
        
        if qualityThreshold is not None:
            self.setQualityThreshold(qualityThreshold)
            
        if normMethod is not None:
            self.setNormMethod(normMethod)
        
        if probThreshold is not None:
            self.setProbThreshold(probThreshold)
                
        self.setQualityCoefficient(qualityCoefficient)
        
        self.setSessionConstant(isMemoryRobot = True, isDBinCSV = True, isMultipleRecognitions = isMultRecognitions, defNumMultRecog = num_mult_recognitions, isSaveRecogFiles = isSaveRecogFiles, isSpeak = False)
        
        if isTestData:
            self.isBNLoaded = True
            self.isBNSaved = True
        """END: set params"""

        stats_openSet, stats_FR, num_recog, FER, num_unknown = self.learnFromFile(db_list=db_list, init_list=init_list, recogs_list=recogs_list,
                                                                                db_file = db_file, init_recog_file = init_recog_file, final_recog_file = final_recog_file, 
                                                                                valid_info_file = valid_info_file, 
                                                                                isSaveImageAn = isSaveImageAn, orig_image_dir = os.path.abspath(os.path.join(recog_folder,"../../../bins")) + "/")

        stats_openSet_percent = self.getDIRFAR(stats_openSet, num_recog, num_unknown)
        stats_FR_percent = self.getDIRFAR(stats_FR, num_recog, num_unknown)
            
        print "time to run: " + str(time.time()-start_time_run)
        return num_recog, FER, stats_openSet_percent, stats_FR_percent, num_unknown


    def runCrossValidationOnRobot(self, num_people, training_folder, test_folder, bin_folder, 
                                 validation_info_file, db_file, 
                                 isTestData = False, db_order_list = None,
                                 isMultRecognitions = False, num_mult_recognitions = None, 
                                 weights = None, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = "evidence", probThreshold = None,
                                 isSaveRecogFiles = True, isSaveImageAn = True):
        
        """Run cross validation offline from validation_info_file"""
        
        start_time = time.time()
        
        """BEGIN: set params"""
        
        self.isSaveRecogFiles = isSaveRecogFiles
        numNoFaceImages = 0
        recog_folder = ""     
        if isTestData:
            recog_folder = test_folder
            if self.isSaveRecogFiles:
                self.resetFilePaths()
                self.setFilePaths(recog_folder)
                self.resetFiles()
                self.resetFilePaths()
                self.copyNetworkDBFromValidation(training_folder, test_folder)
                self.setFilePaths(recog_folder)
            db_new = db_order_list[:]
            self.setUpdateMethod("none")
        else:
            recog_folder = training_folder
            if self.isSaveRecogFiles:
                self.resetFilePaths()
                self.setFilePaths(recog_folder)
                self.resetFiles()
                self.resetFaceDetectionDB()
            db_new = []
            self.setUpdateMethod(updateMethod)
            
        if weights is not None:
            self.setWeights(weights[0], weights[1], weights[2], weights[3], weights[4])
            
        if faceRecogThreshold is not None:
            self.setFaceRecognitionThreshold(faceRecogThreshold)
        
        if qualityThreshold is not None:
            self.setQualityThreshold(qualityThreshold)
            
        if normMethod is not None:
            self.setNormMethod(normMethod)
        
        if probThreshold is not None:
            self.setProbThreshold(probThreshold)
                
        self.setQualityCoefficient(qualityCoefficient)
            
        self.setSessionConstant(isMemoryRobot = True, isDBinCSV = True, isMultipleRecognitions = isMultRecognitions, defNumMultRecog = num_mult_recognitions, isSaveRecogFiles = True, isSpeak = False)

        if isTestData:
            self.isBNLoaded = True
            self.isBNSaved = True
        """END: set params"""
        
        info_file = recog_folder + validation_info_file

        df_info = pandas.read_csv(info_file, converters={"Height": ast.literal_eval, "Time": ast.literal_eval}, usecols ={"N_validation", "Identity", "Original_image", "Bin", "Bin_image", "Validation_image", "Height", "Time"})

        info_list = df_info.values.tolist()
        
        orig_image_list = [x[2] for x in info_list]
        for row in info_list:
            del row[2]
        
        validation_info = [i[1:] for i in info_list]

        if os.path.isfile(db_file):
            try:
                df_db = pandas.read_csv(db_file, dtype={"id": object}, usecols = ["id","name","gender","birthYear","height"])
            except:
                df_db = pandas.read_csv(db_file, dtype={"id": object}, usecols = ["id","name","gender","age","height"])
            db_list = df_db.values.tolist()
        
        num_unknown = 0
        stats_openSet = [0,0] #DIR, FAR
        stats_FR = [0,0]
        num_recog = 0
        for val_recog in validation_info:
            print "num_recog:" + str(num_recog)
            idPersonOrig = val_recog[0]
            print "idPersonOrig:" + str(idPersonOrig)
            numBin = val_recog[1]
            binImage = val_recog[2]
            validationImage = val_recog[3]
            heightPerson = val_recog[4][0]
            timePerson = val_recog[5]
            isRegistered = True
            isAddPersonToDB = False

            if len(db_new) < num_people and idPersonOrig not in db_new:
                isRegistered =  False
                isAddPersonToDB = True
            
            person = []
            if isAddPersonToDB:
                if not os.path.isfile(db_file):
                    warn_msg = db_file + " should exist if a person is to be added to the db (i.e. if isAddPersonToDB is true)."
                    logging.debug(warn_msg)
                    break     
                for i in db_list:
                    if i[0] == str(idPersonOrig):
                        person = i[1:]
                        break
                db_new.append(idPersonOrig)
                person.insert(0, str(len(db_new)))
                person.append([timePerson])
                idPersonNew = person[0]
                isRegistered = False
            else:
                idPersonNew = str(db_new.index(idPersonOrig) + 1)
            print "idPersonNew:" + str(idPersonNew)

            self.setSessionVar(isRegistered = isRegistered, isAddPersonToDB = isAddPersonToDB, personToAdd = person)    

            imagePath = self.imagePath + str(bin_folder) + str(numBin) + "/" + str(binImage) + ".jpg"
            self.setImageToCopy(imagePath)
            self.recog_service.setImagePath(imagePath)
            self.recog_service.setHeightOfPerson(heightPerson, True)
            self.recog_service.setTimeOfPerson(timePerson)
                                               
            identity_est = self.startRecognition() # get the estimated identity from the recognition network
            
            print "identity_est:" + str(identity_est)
                       
            if identity_est == "":
                time.sleep(0.5)
                identity_est = self.startRecognition() # get the estimated identity from the recognition network
                if identity_est == "":
                    numNoFaceImages += 1
                    self.saveImageToTablet(idPersonNew, isDiscardedImage=True, imageName=validationImage)
                    print "discarded image: " + validationImage
                    continue
                
            stats_openSet = self.getPerformanceMetrics(identity_est, idPersonNew, self.unknown_var, isRegistered, stats_openSet)
            
            stats_FR = self.getPerformanceMetrics(self.face_est, idPersonNew, self.unknown_var, isRegistered, stats_FR)
            
            p_id = None
            isRecognitionCorrect = False
            if isRegistered:
                if identity_est != self.unknown_var:
                    if identity_est == idPersonNew:
                        isRecognitionCorrect = True # True if the name is confirmed by the user
                   
            if isRecognitionCorrect:
                self.confirmPersonIdentity(isRobotLearning = isRobotLearning) # save the network, analysis data, csv for learning and picture of the person in the tablet
            else:
                if isAddPersonToDB:
                    p_id = idPersonNew
                    num_unknown += 1
                    self.recog_service.setImagePath(imagePath)
                    self.recog_service.setHeightOfPerson(heightPerson, True)
                    self.recog_service.setTimeOfPerson(timePerson)
                               
                else:
                    p_id = idPersonNew
                self.confirmPersonIdentity(p_id = p_id, isRobotLearning = isRobotLearning)
            num_recog += 1
            time.sleep(0.1)
            print "*"*10

        stats_openSet_percent = self.getDIRFAR(stats_openSet, num_recog, num_unknown)
        stats_FR_percent = self.getDIRFAR(stats_FR, num_recog, num_unknown)
        
        if self.isSaveRecogFiles:
            self.saveBN()
            self.saveConfusionMatrix()
        self.saveFaceDetectionDB()
        
        if self.isDebugMode:
            print "time for validation on robot:" + str(time.time() - start_time) 
        return db_new, stats_openSet_percent, stats_FR_percent, num_recog, numNoFaceImages/(num_recog+numNoFaceImages), num_unknown
    
    #---------------------------------------------PRINT FUNCTIONS---------------------------------------------# 
    
    def printPriors(self):
        """"""
        print "priors:"
        print "I:"
        print self.r_bn.cpt(self.I)[:]
        print "F:"
        print self.r_bn.cpt(self.F)[:]
        print "G:"
        print self.r_bn.cpt(self.G)[:]
#         print "A:"
#         for counter in range(0,len(self.i_labels)):
#             plt.plot(range(self.age_min, self.age_max + 1),self.r_bn.cpt(self.A)[{'I':self.i_labels[counter]}], label=self.i_labels[counter])
#         plt.show()
#         print "H:"
#         for counter in range(0,len(self.i_labels)):
#             plt.plot(range(self.height_min, self.height_max + 1),self.r_bn.cpt(self.H)[{'I':self.i_labels[counter]}], label=self.i_labels[counter])
#         plt.show()
#         print "T:"
#         for counter in range(0,len(self.i_labels)):
#             plt.plot(range(self.time_min, self.time_max + 1),self.r_bn.cpt(self.T)[{'I':self.i_labels[counter]}], label=self.i_labels[counter])
#         plt.show()
    
    def printEvidence(self, face_result, gender_result, age_result, height_result, time_result):
        """"""
        print "face weighted evidence"
        print face_result
        
        print "gender weighted evidence"
        print gender_result
        
#         print "age weighted evidence"
#         plt.plot(range(self.age_min, self.age_max+1),age_result)
#         plt.show() 
# 
#         print "height weighted evidence"
#         plt.plot(range(self.height_min, self.height_max+1),height_result)
#         plt.show()
#         
#         print "time weighted evidence"
#         plt.plot(range(self.time_min, self.time_max+1),time_result)
#         plt.show()
        
    def printInference(self, ie):
        """"""
        print "ie.posterior(self.I):"
        print ie.posterior(self.I)
        print "ie.posterior(self.F):"
        print ie.posterior(self.F)
        print "ie.posterior(self.G):"
        print ie.posterior(self.G)
        print "ie.posterior(self.A):"
        print ie.posterior(self.A)
        print "ie.posterior(self.H):"
        print ie.posterior(self.H)
        print "ie.posterior(self.T):"
        print ie.posterior(self.T)
#         print "ie.posterior(self.A):"
#         plt.plot(range(selfh.age_min, self.age_max+1),ie.posterior(self.A)[:])
#         plt.show()
#         print "ie.posterior(self.H):"
#         plt.plot(range(self.height_min, self.height_max+1),ie.posterior(self.H)[:])
#         plt.show()
#         print "ie.posterior(self.T):"
#         plt.plot(range(self.time_min, self.time_max+1),ie.posterior(self.T)[:])
#         plt.show()
    
    def printDB(self):
        """"""
        print "database:"
        print "self.i_labels: " + str(self.i_labels)
        print "self.genders: "  + str(self.genders)
        print "self.ages: " + str(self.ages)
        print "self.heights: " + str(self.heights)
        print "self.times: " + str(self.times)
        print "self.num_people: " + str(self.num_people)
                        
if __name__ == "__main__":

    RB = RecogniserBN()

