#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Analysis module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): S.Bera. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-10-04
# Last update: 2024-12-14

#libs
import time
import struct
import os, time, glob

import numpy as np
np.seterr(all='raise')
from numpy.linalg import norm

from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.analysis.AQiPT_analysis_utils import *
from AQiPT.modules.emulator.AQiPTemulator import bitCom

from scipy.linalg import solve
import scipy.optimize

import pandas as pd
import h5py

import matplotlib.pyplot as plt

from astropy.io import fits



#format name files cameras

#andor
andor = 'Andor';
ANDOR_Abs_file_name = "R*_AndorAbs.fts";
ANDOR_Abs2_file_name = "R*_AndorAbs2.fts";
ANDOR_Div_file_name = "R*_AndorDiv.fts";
ANDOR_Bgd_file_name = "R*_AndorBgd.fts";
ANDOR_FILE_NAMES = [ANDOR_Abs_file_name, ANDOR_Div_file_name, ANDOR_Bgd_file_name];

#ids
ids = 'IDS';
IDS_Abs_file_name = "R*_Abs1.fts";
IDS_Div_file_name = "R*_Div1.fts";
IDS_Bgd_file_name = "R*_Bgd.fts";
IDS_FILE_NAMES = [IDS_Abs_file_name, IDS_Div_file_name, IDS_Bgd_file_name];


#data structures
SIDE_IMAGING_FILE = 'data1.csv';
TOP_IMGAGING_FILE = 'data2.csv';


###################################################################################################
#######################                 Frontend Analysis                 #########################
###################################################################################################

#####################################################################################################
#DataManager AQiPT class
#####################################################################################################


class DataManager:
    def __init__(self,  directory=aqipt.directory.data_depository_dir, filename='default_data_'+time.strftime("%Y-%m-%d_%Hh%Mm"), comments='Default comments',authors= 'Data Manager by AQiPT.',scan_variables=None):

        self.filename = filename;
        self.directory = directory+filename+"/";
        self._variables = scan_variables;

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with h5py.File(self.directory +'Processed_data.hdf5', 'w-') as raw_data_file:
            _main_group = raw_data_file.create_group(self.filename+'_processed');
            # _first_dataset = _main_group.create_dataset("Dataset_0", data=[0,1,2,3,4]); #not necessary but useful to debug
        self.Processed_data = h5py.File(self.directory +'Processed_data.hdf5', 'r+');

        with h5py.File(self.directory +'RAW_data.hdf5', 'w-') as processed_data_file:
            _main_group = processed_data_file.create_group('Dataset_raw_'+time.strftime("%Y-%m-%d_%Hh%Mm")); #main group within RAW_data.hdf5
            _image_group = _main_group.create_group("Image");
            _text_group  = _main_group.create_group("Text");
            _table_group = _main_group.create_group("Table");
            _array_group = _main_group.create_group("Array");
            _aqipt_group = _main_group.create_group("AQiPT objects");

            # _first_dataset = _main_group.create_dataset("Dataset_0", data=[0,1,2,3,4]); #not necessary but useful to debug
        self.RAW_data = h5py.File(self.directory +'RAW_data.hdf5', 'r+');


        self._date = time.strftime("%Y-%m-%d_%Hh%Mm");
        self._comments = comments;
        self._authors = authors;

        self._groups = [group for group in self.RAW_data.keys()];

    def getPath(self):
        return (self.directory)

    def printPath(self):
        print(self.directory)

    def getVariable(self):
        return self.variable

    def add_RAW_data(self, new_data, group_label:str, subgroup_label:str, data_label:str):

        if group_label in self._groups: #check group
            if subgroup_label in [_subgroup for _subgroup in self.RAW_data[group_label].keys()]: #check dataset
                self.RAW_data[group_label][subgroup_label].create_dataset(data_label, data=new_data); #set new raw data (dataset) in one of the premade groups
                print('New data added!')

    def remove_data(self, data):
        self.data.remove(data);

class ArrayNNClassifier:

    '''
        Class for atom detection in tweezer arrays. It does train a NN with a set of training images and test it with
        other set, afterwards the NN is trained and using the tensorflow model it can be used to classify what configuration
        is in the array.

        Note: tests was carried with 10k images training set and tested with 50 with accuracy >90%

        Attributes:

        keys_to_features (dict): dictionary with figures and keywords association
        parsed_features (tensorflow.parser): tensorflow parser
        image (tensorflow.tensor): image (buffer)
        images (array): loaded images (buffer)
        labels (array): loaded labels of images (buffer)

        path2dataset (str): path to dataset of images
        training_folder (str): folder of training images for the model
        test_folder (str): folder of the test images

        raw_dataset (tensoflow.data): tensorflow data object
        parsed_image_dataset (array): image dataser after parser passed

        training_images (array): set of training images
        label_training_images (list(str)):
        test_images (array): set of test images
        label_test_images (list(str)):
        
        model (tensorflow.model): neural network tensorflow model 
        optimizer (str): tensorflow optimizer
        loss (str): type of loss considered in the model
        metrics (list(str)): metrics used to train the model e.g., 'accuracy'
        history (tensorflow.fit): tensorflow history of the model

        fig_ext (str): file of images extension

        test_loss (float):
        test_accuracy (float):


        Methods:

        _parse_function: parser of function for getting raw datasets

        import_images: get images
        load_images_and_labels: get images from folder/directory
        load_labels: get labels from folder/directory
        load_tensorflow_dataset: get images from tensorflow dataser 

        get_classes: get classes of images to classify
        check_trained_images: check model with trained images
        binary_to_int: transform binary number into integer
        set_model: define NN tensorflow model for analysis
        compile_model: compile the loaded tensorflow model

        fit_model: fit model to training and test datasets
        evaluate: evaluate model
        get_classification: check model for classification
    '''

    def __init__(self, training_folder, test_folder, path2dataset=None, fig_ext='.png', optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):

        self.keys_to_features = None
        self.parsed_features = None
        
        self.image = None
        self.images = []
        self.labels = []
        
        self.path2dataset = None
        self.training_folder = training_folder
        self.test_folder = test_folder

        self.raw_dataset = None
        self.parsed_image_dataset = None

        self.training_images = None
        self.label_training_images = None
        self.test_images = None
        self.label_test_images = None

        self.model = None
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.history = None

        self.fig_ext = fig_ext
        
        self.test_loss = 0
        self.test_accuracy = 0

    def _parse_function(self, proto):
        '''
           Function to parse the image from TFRecord
        '''
        
        self.keys_to_features = {'image_raw': tf.io.FixedLenFeature([], tf.string)} #feature description dictionary
        
        self.parsed_features = tf.io.parse_single_example(proto, self.keys_to_features) #parser
       
        self.image = tf.io.parse_tensor(self.parsed_features['image_raw'], out_type=tf.float32)  #decode image
        self.image = tf.reshape(self.image, (250, 250, 1))  #reshape to the original size

        return self.image

    def import_images(self):
        '''
            Load all training and test images for NN
        '''

        self.training_images, self.label_training_images = self.load_images_and_labels(self.training_folder)  #read training images and labels
        self.test_images, self.label_test_images = self.load_images_and_labels(self.est_folder) #read test images and labels

        return (self.training_images, self.label_training_images), (self.test_images, self.label_test_images)

    def load_images_and_labels(self, folder, container='images', fname='metadata.txt'):

        '''
            Image import function from folder with .txt file with one column with the label e.g., 1011 and image files
            the tensorflow file e.g., my_dataset.tfrecord
        '''

        #include paths
        images_folder = os.path.join(folder, container)
        metadata_file = os.path.join(folder, fname)

        #read metadata (labels)
        with open(metadata_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.image_files = os.listdir(images_folder) #read and sort images based on "shot_X" where X is a number

        #helper function to extract X value from file name
        def extract_x_value(filename):

            match = re.match(r"shot_(\d+)_\d+"+self.fig_ext, filename)

            if match:
                return int(match.group(1))  #extract X as an integer
            return float('inf')  # Fallback for unexpected file names

        self.sorted_image_files = sorted(self.image_files, key=extract_x_value) #sort image files by the X value

        #read images
        for image_file in self.sorted_image_files:

          img_path = os.path.join(images_folder, image_file)
          img = Image.open(img_path).convert('RGB')  #convert to RGB if needed
          img = np.array(img)  #convert to numpy array
          self.images.append(img)

        #convert images and labels to numpy arrays for easier processing
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        return self.images, self.labels

    def load_labels(self, folder, fname='metadata.txt'):
        '''
            Load labels from .txt file in folder
        '''

        metadata_file = os.path.join(folder, fname) #include in paths

        #get metadata (labels)
        with open(metadata_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        return np.array(self.labels)

    def load_tensorflow_dataset(self, new_path2dataset=None):
        '''
            Image import function from the tensorflow file e.g., my_dataset.tfrecord
        '''

        if new_path2dataset!=None:
            self.path2dataset = new_path2dataset

        self.raw_dataset = tf.data.TFRecordDataset(self.path2dataset) #load TFRecord file        
        self.parsed_image_dataset = self.raw_dataset.map(self._parse_function) #parser of data

        return np.array([_image.numpy() for _image in self.parsed_image_dataset])

    def get_classes(self, N):
        '''
            Get classes of possible combinations of atoms present in the array
        '''
        return [item for sublist in [bitCom(N, idx) for idx in range(N + 1)] for item in sublist]

    def check_trained_images(self, train_labels, figure_size=(10, 10), nrsamples=10):
        '''
            Check trained images with labels
        '''
        
        fig, axes = plt.subplots(1, nrsamples, figsize=figure_size)
        
        for i in range(nrsamples):

            image = self.training_images[i]
            denormalized_image = (image + 1) / 2
            axes[i].imshow(denormalized_image)
            axes[i].set_title(train_labels[i])
            axes[i].axis('off')   

    def binary_to_int(self, binary_str):
        '''
            Convert binary strings to integers
        '''
        return int(binary_str, 2)

    def set_model(self):
        '''
            Euristic NN architecture classifying objects, specifically for circular objects, it is
            designed to classify an image with an output of 2^4 possible outcomes, related to the possibility
            of 2x2 arrays of loaded atoms. 
        '''
        self.model = models.Sequential([layers.Conv2D(64, (3, 3), activation='relu', input_shape=(250, 250, 1)),
                                        layers.MaxPooling2D((2, 2), strides=(2, 2)),
                                        layers.Conv2D(128, (3, 3), activation='relu'),
                                        layers.MaxPooling2D((2, 2), strides=(2, 2)),
                                        layers.Flatten(),
                                        layers.Dense(120, activation='relu'),
                                        layers.Dense(84, activation='relu'),
                                        layers.Dense(16, activation='softmax')
                                      ])

        self.model_summary = self.model.summary()
        print(self.model_summary)

        return self.model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def fit_model(self):
        self.history = self.model.fit(self.training_images, self.label_training_images, epochs=16, validation_data=(self.test_images, self.label_test_images))

    def evaluate(self):

        self.test_loss, self.test_accuracy = self.model.evaluate(self.test_images, self.test_labels)
        print(f'Accuracy of the neural network on the {self.test_images.shape[0]} test images: {self.test_accuracy * 100:.2f}%')

    def get_classification(self, image, probabilities):
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)

        denormalized_image = (image + 1) / 2

        ax1.imshow(denormalized_image)
        ax1.axis('off')
        ax2.barh(np.arange(16), probabilities)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(16))
        ax2.set_yticklabels(classes)
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()


###################################################################################################
#######################                 Middleware Analysis               #########################
###################################################################################################

#####################################################################################################
#Data AQiPT class
#####################################################################################################
def openFTS(file):

    '''
        Open .fts file and yields 2D array

        INPUTS:
        -------

        file (str): .fts file name
        
        OUTPUTS:
        --------

        (ndarray): 2D array python object

    '''

    _file_load = fits.open(file);
    return _file_load[0].data

def importImages(path, filenames):

    Abs_file_name, Div_file_name, Bgd_file_name = filenames;

    os.chdir(path);
    
    _Abs = glob.glob(os.path.join(path, Abs_file_name));
    _Div = glob.glob(os.path.join(path, Div_file_name));
    _Bgd = glob.glob(os.path.join(path, Bgd_file_name));

    _dimensions = np.shape(openFTS(_Abs[0]));

    _x = _dimensions[0]; _y = _dimensions[1];

    _images_list = {"Absorption": np.empty([_x, _y, len(_Abs)]), "Division": np.empty([_x, _y, len(_Abs)]),"Background": np.empty([_x, _y, len(_Abs)])}

    for i in range(len(_Abs)):
        _images_list["Absorption"][:, :, i] = openFTS(_Abs[i]);
        _images_list["Division"][:, :, i] = openFTS(_Div[i]);
        _images_list["Background"][:, :, i] = openFTS(_Bgd[i]);

    return _images_list


class Data:

    '''
        AQiPT class for data types usually instantiated as Image,Table,Text,Array or AQiPT data.

    '''
    def __init__(self, raw_data=None, path2raw_data=None):

        if raw_data!=None:
            self.raw_data = raw_data
        elif path2raw_data!=None:
            self._rawData_directory = path2raw_data;


        self.versions = [raw_data]  # Store the initial raw data as the first version

    def analyze(self, analysis_type):
        # Perform analysis on the raw data and store the new version
        # Update the versions list accordingly
        new_data_version = perform_analysis(self.raw_data, analysis_type)
        self.versions.append(new_data_version)

    def get_latest_version(self):
        return self.versions[-1]

    def get_all_versions(self):
        return self.versions

class ImageData(Data):

    def __init__(self, raw_data, path2raw_data, subclass_attribute, cameraType='IDS'):
        super().__init__(parent_attribute);

        self._camera_type = cameraType;

        if self._camera_type=='IDS':
           self. __image_filename = IDS_FILE_NAMES;

        elif self._camera_type=='Andor':
            self.__image_filename = ANDOR_FILE_NAMES;

        else:
            self.__image_filename = None;

        try:
            self._image_list = importImages(self.path2raw_data, self.__image_filename);
        except:
            self._image_list = None;

    def process(self):
        pass

    def analyze(self, analysis_type):
        # Perform image-specific analysis
        if analysis_type == 'peaks':
            # Implement peak finding for image data
            pass
        elif analysis_type == 'fitting':
            # Implement fitting analysis for image data
            pass
        else:
            raise ValueError("Invalid analysis type for image data.")

class TableData(Data):

    def __init__(self, raw_data, path2raw_data, subclass_attribute, cameraType='IDS'):
        super().__init__(parent_attribute);

        self._camera_type = cameraType;

        if self._camera_type=='IDS':
           self. __datafile_name = IDS_FILE_NAMES;

        elif self._camera_type=='Andor':
            self.__datafile_name = ANDOR_FILE_NAMES;

        else:
            self.__datafile_name = None;

    def analyze(self, analysis_type):
        # Perform table-specific analysis
        if analysis_type == 'peaks':
            # Implement peak finding for table data
            pass
        elif analysis_type == 'fitting':
            # Implement fitting analysis for table data
            pass
        elif analysis_type == 'NN':
            # Implement fitting analysis for array of atoms
            pass
        else:
            raise ValueError("Invalid analysis type for table data.")

class TextData(Data):
    pass

class ArrayData(Data):
    pass

class AQiPTData(Data):
    pass


###################################################################################################
#######################                 Backend Analysis                  #########################
###################################################################################################

#####################################################################################################
#Trc AQiPT class
#####################################################################################################
class Trc:
    _recTypes = (
        "single_sweep", "interleaved", "histogram", "graph",
        "filter_coefficient", "complex", "extrema",
        "sequence_obsolete", "centered_RIS", "peak_detect"
    )
    _processings = (
        "no_processing", "fir_filter", "interpolated", "sparsed",
        "autoscaled", "no_result", "rolling", "cumulative"
    )
    _timebases = (
        '1_ps/div', '2_ps/div', '5_ps/div', '10_ps/div', '20_ps/div',
        '50_ps/div', '100_ps/div', '200_ps/div', '500_ps/div', '1_ns/div',
        '2_ns/div', '5_ns/div', '10_ns/div', '20_ns/div', '50_ns/div',
        '100_ns/div', '200_ns/div', '500_ns/div', '1_us/div', '2_us/div',
        '5_us/div', '10_us/div', '20_us/div', '50_us/div', '100_us/div',
        '200_us/div', '500_us/div', '1_ms/div', '2_ms/div', '5_ms/div',
        '10_ms/div', '20_ms/div', '50_ms/div', '100_ms/div', '200_ms/div',
        '500_ms/div', '1_s/div', '2_s/div', '5_s/div', '10_s/div',
        '20_s/div', '50_s/div', '100_s/div', '200_s/div', '500_s/div',
        '1_ks/div', '2_ks/div', '5_ks/div', 'EXTERNAL'
    )
    _vCouplings = ('DC_50_Ohms', 'ground', 'DC_1MOhm', 'ground', 'AC,_1MOhm')
    _vGains = (
        '1_uV/div', '2_uV/div', '5_uV/div', '10_uV/div', '20_uV/div',
        '50_uV/div', '100_uV/div', '200_uV/div', '500_uV/div', '1_mV/div',
        '2_mV/div', '5_mV/div', '10_mV/div', '20_mV/div', '50_mV/div',
        '100_mV/div', '200_mV/div', '500_mV/div', '1_V/div', '2_V/div',
        '5_V/div', '10_V/div', '20_V/div', '50_V/div', '100_V/div',
        '200_V/div', '500_V/div', '1_kV/div'
    )

    def __init__(self):
        '''
        use trc.open(fName) to open a Le Croy .trc file
        '''
        self._f = None
        # offset to start of WAVEDESC block
        self._offs = 0
        self._smplFmt = "int16"
        self._endi = ""

    def open(self, fName):
        '''
            _readS .trc binary files from LeCroy Oscilloscopes.
            Decoding is based on LECROY_2_3 template.
            [More info]
            (http://forums.ni.com/attachments/ni/60/4652/2/LeCroyWaveformTemplate_2_3.pdf)

            Parameters
            -----------
            fName = filename of the .trc file

            Returns
            -----------
            a tuple (x, y, d)

            x: array with sample times [s],

            y: array with sample  values [V],

            d: dictionary with metadata

            M. Betz 09/2015
        '''
        with open(fName, "rb") as f:
            # Binary file handle
            self._f = f
            self._endi = ""
            temp = f.read(64)
            # offset to start of WAVEDESC block
            self._offs = temp.find(b'WAVEDESC')

            # -------------------------------
            #  Read WAVEDESC block
            # -------------------------------
            # Template name
            self._TEMPLATE_NAME = self._readS("16s", 16)
            if self._TEMPLATE_NAME != "LECROY_2_3":
                print(
                    "Warning, unsupported file template:",
                    self._TEMPLATE_NAME,
                    "... trying anyway"
                )
            # 16 or 8 bit sample format?
            if self._readX('H', 32):
                self._smplFmt = "int16"
            else:
                self._smplFmt = "int8"
            # Endian-ness ("<" or ">")
            if self._readX('H', 34):
                self._endi = "<"
            else:
                self._endi = ">"
            #  Get length of blocks and arrays
            self._lWAVE_DESCRIPTOR = self._readX("l", 36)
            self._lUSER_TEXT = self._readX("l", 40)
            self._lTRIGTIME_ARRAY = self._readX("l", 48)
            self._lRIS_TIME_ARRAY = self._readX("l", 52)
            self._lWAVE_ARRAY_1 = self._readX("l", 60)
            self._lWAVE_ARRAY_2 = self._readX("l", 64)

            d = dict()  # Will store all the extracted Metadata

            # ------------------------
            #  Get Instrument info
            # ------------------------
            d["INSTRUMENT_NAME"] = self._readS("16s", 76)
            d["INSTRUMENT_NUMBER"] = self._readX("l", 92)
            d["TRACE_LABEL"] = self._readS("16s", 96)

            # ------------------------
            #  Get Waveform info
            # ------------------------
            d["WAVE_ARRAY_COUNT"] = self._readX("l", 116)
            d["PNTS_PER_SCREEN"] = self._readX("l", 120)
            d["FIRST_VALID_PNT"] = self._readX("l", 124)
            d["LAST_VALID_PNT"] = self._readX("l", 128)
            d["FIRST_POINT"] = self._readX("l", 132)
            d["SPARSING_FACTOR"] = self._readX("l", 136)
            d["SEGMENT_INDEX"] = self._readX("l", 140)
            d["SUBARRAY_COUNT"] = self._readX("l", 144)
            d["SWEEPS_PER_ACQ"] = self._readX("l", 148)
            d["POINTS_PER_PAIR"] = self._readX("h", 152)
            d["PAIR_OFFSET"] = self._readX("h", 154)
            d["VERTICAL_GAIN"] = self._readX("f", 156)
            d["VERTICAL_OFFSET"] = self._readX("f", 160)
            # to get floating values from raw data:
            # VERTICAL_GAIN * data - VERTICAL_OFFSET
            d["MAX_VALUE"] = self._readX("f", 164)
            d["MIN_VALUE"] = self._readX("f", 168)
            d["NOMINAL_BITS"] = self._readX("h", 172)
            d["NOM_SUBARRAY_COUNT"] = self._readX("h", 174)
            # sampling interval for time domain waveforms
            d["HORIZ_INTERVAL"] = self._readX("f", 176)
            # trigger offset for the first sweep of the trigger,
            # seconds between the trigger and the first data point
            d["HORIZ_OFFSET"] = self._readX("d", 180)
            d["PIXEL_OFFSET"] = self._readX("d", 188)
            d["VERTUNIT"] = self._readS("48s", 196)
            d["HORUNIT"] = self._readS("48s", 244)
            d["HORIZ_UNCERTAINTY"] = self._readX("f", 292)
            d["TRIGGER_TIME"] = self._getTimeStamp(296)
            d["ACQ_DURATION"] = self._readX("f", 312)
            d["RECORD_TYPE"] = Trc._recTypes[
                self._readX("H", 316)
            ]
            d["PROCESSING_DONE"] = Trc._processings[
                self._readX("H", 318)
            ]
            d["RIS_SWEEPS"] = self._readX("h", 322)
            d["TIMEBASE"] = Trc._timebases[self._readX("H", 324)]
            d["VERT_COUPLING"] = Trc._vCouplings[
                self._readX("H", 326)
            ]
            d["PROBE_ATT"] = self._readX("f", 328)
            d["FIXED_VERT_GAIN"] = Trc._vGains[
                self._readX("H", 332)
            ]
            d["BANDWIDTH_LIMIT"] = bool(self._readX("H", 334))
            d["VERTICAL_VERNIER"] = self._readX("f", 336)
            d["ACQ_VERT_OFFSET"] = self._readX("f", 340)
            d["WAVE_SOURCE"] = self._readX("H", 344)
            d["USER_TEXT"] = self._readS(
                "{0}s".format(self._lUSER_TEXT),
                self._lWAVE_DESCRIPTOR
            )

            y = self._readSamples()
            y = d["VERTICAL_GAIN"] * y - d["VERTICAL_OFFSET"]
            x = np.arange(1, len(y) + 1, dtype=float)
            x *= d["HORIZ_INTERVAL"]
            x += d["HORIZ_OFFSET"]
        self.f = None
        self.x = x
        self.y = y
        self.d = d
        return x, y, d

    def _readX(self, fmt, adr=None):
        ''' extract a byte / word / float / double from the binary file f '''
        fmt = self._endi + fmt
        nBytes = struct.calcsize(fmt)
        if adr is not None:
            self._f.seek(adr + self._offs)
        s = struct.unpack(fmt, self._f.read(nBytes))
        if(type(s) == tuple):
            return s[0]
        else:
            return s

    def _readS(self, fmt="16s", adr=None):
        ''' read (and decode) a fixed length string '''
        temp = self._readX(fmt, adr).split(b'\x00')[0]
        return temp.decode()

    def _readSamples(self):
        # ------------------------
        #  Get main sample data with the help of numpys .fromfile(
        # ------------------------
        # Seek to WAVE_ARRAY_1
        self._f.seek(
            self._offs + self._lWAVE_DESCRIPTOR +
            self._lUSER_TEXT + self._lTRIGTIME_ARRAY +
            self._lRIS_TIME_ARRAY
        )
        y = np.fromfile(self._f, self._smplFmt, self._lWAVE_ARRAY_1)
        if self._endi == ">":
            y.byteswap(True)
        return y

    def _getTimeStamp(self, adr):
        ''' extract a timestamp from the binary file '''
        s = self._readX("d", adr)
        m = self._readX("b")
        h = self._readX("b")
        D = self._readX("b")
        M = self._readX("b")
        Y = self._readX("h")
        trigTs = datetime.datetime(
            Y, M, D, h, m, int(s), int((s - int(s)) * 1e6)
        )
        return trigTs

    def fromSPREADSHEET(self, fname_lst:list):
        frames_lst = []
        for fname in fname_lst:
            pframe = pd.read_excel(open(fname,'rb'), 
                                   header=None,
                                   sheet_name='Sheet1')
            pframe.columns= ['dRedProbe', 'Integrate Ion signal']
            frames_lst.append(pframe)

def fromTRC(filename, Ntrace, roi):
    '''
    \example
    dataset = importTr('C2--trace_data.trc', 200, (8500, 11000))

    '''
        # trc = Trc()
    datX, datY, m = Trc().open(filename)
    datY2 = datY.reshape(Ntrace, int(datY.size/Ntrace))
    datY2 = -datY2[:, roi[0]:roi[1]]
    return datY2

def fromPRN(filename, delim=',', nr_cols=2):
    '''
        xs, ys = fromPrn('csa.prn')
    '''
    metadata = np.genfromtxt(filename, delimiter=delim, usecols=np.arange(0,nr_cols))
    metadata = metadata[2:]
    xValues=[]; yValues=[];
    for i in range(len(metadata)):
        xValues.append(metadata[i][0])
        yValues.append(metadata[i][1])
    return np.asarray(xValues), np.asarray(yValues)

def fromCSV(filename, pathDir=None):
    df = pd.read_csv(filename, header=None, converters={i: str for i in range(50)})
    return df[0]

def findpeaksOMP(dataset,roi,w,Dbg=np.array([]),maxpeaks=30,thresh=0.05,tolx=0.01, makeplot=False,savefig=False):
    '''
        Robust peak detection using Orthogonal Matching Pursuit (OMP) v1.0 S. Whitlock, December 6 2020
        Args:
          dataset (M,T float array): array of time traces of shape (M, T)
          roi: (float,float) boundaries defining the region of interest where peaks should be found (
          w: (float) Gaussian width of the peaks (also sets the minimum time resolution between peaks)
          Dbg (N,T float array): array of background time traces of shape (N,T) for noise minimization (optional)
          maxpeaks (int): number of non-zero coefficients used in OMP to decompose the data. This constrains the number of peaks that can be found
          thresh (float): threshold amplitude for a peak detection event
          tolx (float < 1): solver tolerance
          makeplot (bool): generate a plot
          savefig (bool): save plot to file peakfits.pdf
        Returns:
          Number of peaks detected in each time trace; vector of length M
    '''
    (M,T)=dataset.shape #dimensions of the dataset
        
    # generate dictionary of gaussian peaks (atoms)
    t0 = round(T/2)
    dt=w/2
    t = np.arange(0,T)
    atom = np.exp(-(t-t0)**2/(2*w**2)); atom =atom/np.sum(atom)
    
    centers = np.arange(roi[0],roi[1]+dt,dt)
    D = np.zeros((T,centers.size)) # Pre-allocate matrix
    for j in np.arange(0,centers.size):
        shift = int((centers[j]-t0))
        D[:,j] = np.roll(atom,shift) 
        
    # include background region into dictionary
    if Dbg.size>0:
        Dplus =np.append(D,Dbg.T,axis=1)
    else:
        Dplus=D
    
    #prepare plot
    if makeplot:
        my_dpi=300
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 20), dpi=my_dpi)
        # ax1.set_aspect(1000)
        # ax2.set_aspect(1000)
        fig.tight_layout()
    
    #interate over dataset and perform orthogonalmatchingpursuits
    out=[]
    for k in np.arange(0,M-1):
        data=dataset[k,:]
        x = OrthogonalMP(Dplus, data, nnz=maxpeaks, tol=tolx, positive=True)
        s = x[0:D[1,:].size]        
        
        #apply threshold condition
        amplitudes = np.max(D * s,0)
        numberofpeaks=np.sum(amplitudes>thresh)

        if makeplot and k< 100:
            s[amplitudes<thresh]=0
            peaks = sum((D * s).T)
            reconstruction = Dplus @ x
            ax1.plot(t,data-20*thresh*k,t,reconstruction-20*thresh*k,'-k')
            ax2.plot(t,data-20*thresh*k,t,peaks-20*thresh*k,'-k')
            
            ax1.plot([roi[0],roi[0]],[-20*thresh*k,-20*thresh*(k-1)],':r')#region of interest
            ax1.plot([roi[1],roi[1]],[-20*thresh*k,-20*thresh*(k-1)],':r')
            ax2.plot([roi[0],roi[0]],[-20*thresh*k,-20*thresh*(k-1)],':r')#region of interest
            ax2.plot([roi[1],roi[1]],[-20*thresh*k,-20*thresh*(k-1)],':r')

        out.append(numberofpeaks)
    # if savefig:
    #     fig.savefig("peakfits.pdf")

    return np.array(out)

