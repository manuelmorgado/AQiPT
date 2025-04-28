#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Analysis module

# Author(s): EQM GROUP
#            Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
#            Swayangdipta Bera. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): 
# Created: 2021-10-04
# Last update: 2024-12-14

import numpy as np
import matplotlib.pyplot as plt

std_params = {'cool_double_3d' : {'a': -17.529, 'b': 444.089},
              'cool_single_3d' :220,
              'cool_single_2d' : 200,
              'cool_double_2d' : {'a': -19.279, 'b': 451.928},
              'D1_spec' : {'a': -6.334, 'b': 225.785},
              'D2_spec' : {'a': -9.651, 'b': 232.330},
              'repumper_double' : {'a': -17.358, 'b': 445.961},
              'repumper_single' : -214.3,
              'pusher' : {'a': -8.994, 'b': 226.633},
              'Abs_img' : {'a': -9.059, 'b': 230.524},
              'abs_img_repump' : {'a': -8.99, 'b': 227.85},
              'sp_after_crossover_D1' : 219.9,
              'crossover' : 461.7/2,
              'hyperfine_split' : 461.7};


'''
    G: gain
    S: 
    QE: quantum efficiency
    bias: bias
    N_ADC:
    pixels:total pixels we are integrating
    N_ph: number of photons?    
'''
camera_params = {'G': 200, 'S': 17, 'QE': 0.85, 'bias': 500}


def gaussian2D(vars, amp, x0, y0, sigma_x, sigma_y):
    x = vars[:, 0]
    y = vars[:, 1]
        # a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        # b = (np.sin(2*theta))/(2*sigma_x**2) - (np.sin(2*theta))/(2*sigma_y**2)
        # c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        # return amp*np.exp(-(a*((x - x0)**2) + 2*b*(x - x0)*(y - y0) + c*((y - y0)**2)))

    _gauss2D = amp*np.exp(-(x - x0)**2 / (2*sigma_x**2)-(y - y0)**2 / (2*sigma_y**2));
    return _gauss2D

def gaussian1D(x, amp, x0, sigma_x):
    _gauss1D = amp*np.exp(-(x - x0)**2 / (2*sigma_x**2));
    return _gauss1D

def calculate_optimal_bins(data):
    '''
        Calculate the optimal number of bins for a histogram using the Freedman-Diaconis rule.
        
        Args:
        - data: A list or NumPy array containing the data points.
        
        Returns:
        - num_bins: The optimal number of bins as an integer.
    '''
    # Step 1: Calculate the Interquartile Range (IQR)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Step 2: Determine the number of data points (n)
    n = len(data)

    # Step 3: Calculate the optimal bin width using the Freedman-Diaconis rule
    bin_width = 2 * IQR / (n ** (1/3))

    # Step 4: Calculate the number of bins
    data_range = max(data) - min(data)
    num_bins = int(data_range / bin_width)

    # Step 5: Round the result to the nearest integer
    num_bins = round(num_bins)

    return num_bins

def create_animated_gif(image_list, output_gif, frame_duration=100, loop=True):
    '''
        Create an animated GIF from a list of 2D images, which are NumPy arrays.

        Args:
        - image_list: A list of 2D images as NumPy arrays.
        - output_gif: The file name for the output animated GIF.
        - frame_duration: The duration (in milliseconds) for each frame in the GIF. Default is 100 milliseconds.
        - loop: Whether the GIF should loop indefinitely. Default is True.

        Returns:
        - None
    '''
    # Convert NumPy arrays to PIL Image objects
    image_list_pil = [Image.fromarray(image) for image in image_list]

    # Save the list of images as an animated GIF
    image_list_pil[0].save(
        output_gif,
        save_all=True,
        append_images=image_list_pil[1:],
        duration=frame_duration,
        loop=0 if loop else 1  # Loop indefinitely (0 means loop forever)
    )

    # Check if the GIF was created successfully
    if os.path.exists(output_gif):
        print(f"Animated GIF '{output_gif}' created successfully.")
    else:
        print("Failed to create the animated GIF.")

def s_threshold(sigmaH, sigmaL, SH, SL):
    return (sigmaH*SL + sigmaL*SH)/(sigmaH+sigmaL)

def double_gaussian(f, arg_gaussian1, arg_gaussian2):
    _gaussian_1 = gaussian1D(arg_gaussian1['x'], arg_gaussian1['amp'], arg_gaussian1['x0'], arg_gaussian1['sigma_x']);
    _gaussian_2 = gaussian1D(arg_gaussian2['x'], arg_gaussian2['amp'], arg_gaussian2['x0'], arg_gaussian2['sigma_x']);

    return (1-f)*_gaussian_1 + f*_gaussian_2

def int_threshold(f, data, t):

    _threshold = (1-f)*(np.sum(data[:t])) + f*(np.sum(data[t:]));

    return _threshold

def ecdf(a):
    x, counts = np.unique(a, return_counts=True);
    _cusum = np.cumsum(counts);
    return x, _cusum/_cusum[-1]

def map_to_256_values(array):
    '''
    Map the values of a 2D NumPy array to the range [0, 255].

    Args:
    - array: The input 2D NumPy array.

    Returns:
    - A new 2D NumPy array with values in the range [0, 255].
    '''
    # Find the minimum and maximum values in the array
    min_value = np.min(array)
    max_value = np.max(array)

    # Map the values to the range [0, 255]
    mapped_array = 255 * (array - min_value) / (max_value - min_value)
    
    # Round and cast the values to integers
    mapped_array = np.round(mapped_array).astype(np.uint8)

    return mapped_array

def FitScanResult(fit_fn, ScanVar, ydata, ydata_std, ini_guess):
    output = {}
    xdata = ScanVar["Values"]
    popt, cov = scipy.optimize.curve_fit(fit_fn, xdata, ydata, ini_guess) #fn, xdata, ydata, initial guess
    fig = plt.figure()
    x_var = np.linspace(np.min(xdata), np.max(xdata), 151)
    BestFit = fit_fn(x_var, *popt)
    plt.errorbar(xdata, ydata, ydata_std, color = 'b', fmt = 'o')
    plt.plot(x_var, BestFit, '-r', linewidth = 2)
    plt.xlabel(ScanVar["Name"])
    plt.savefig('Plot_scan_result_fit.jpg', dpi = 300)
    
    output["x"] = x_var
    output["y_fit"] = BestFit
    output["popt"] = popt
    return output, fig

def size_to_AN(N):
    '''Set figure dimensions to AN. Width denotes the larger dimension.
    '''
    if N%2 == 0:
        height_mm = 1/2**(N/2) * 1000/2**(1/4)
        width_mm = 1/2**(N/2) * 10002**(1/4)
    else:
        height_mm = 1/2**((N+1)/2) * 10002**(1/4)
        width_mm = 1/2**((N-1)/2) * 1000/2**(1/4)
        width_in = 0.0393701 * width_mm
        height_in = 0.0393701 * height_mm
    return (width_in, height_in)

#this should be a method for ImportData
def AnalysisReport(path):
    os.chdir(path)
    lstImg = glob.glob(os.path.join(path, '*.png'))
    nImg = len(lstImg)
    if nImg == 1: 
        img = mpimg.imread(lstImg[0])
        fig, ax = plt.subplots(1,1, figsize = size_to_AN(6), dpi = 300)
        ax.imshow(img)
        ax.axis('off') 
        plt.suptitle(path, fontsize = 6)
        plt.savefig('all image.jpeg', dpi = 300)
    
    elif nImg == 2:
        fig, ax = plt.subplots(1,2, figsize = size_to_AN(6),dpi = 300)
        plt.axis('off')
        for i in range(nImg):
            img = mpimg.imread(lstImg[i])
            ax[i].imshow(img)
        plt.suptitle(path, fontsize = 6)
        plt.savefig('all image.jpeg', dpi = 300)

    elif nImg == 3:
        fig, ax = plt.subplots(2,2, figsize = size_to_AN(6),dpi = 300)

        img0 = mpimg.imread(lstImg[0])
        ax[0][0].imshow(img0)
        ax[0][0].axis('off')

        img1 = mpimg.imread(lstImg[1])
        ax[0][1].imshow(img1)
        ax[0][1].axis('off')

        img2 = mpimg.imread(lstImg[2])
        ax[1][0].imshow(img2)
        ax[1][0].axis('off')

        ax[1][1].axis('off')

        plt.tight_layout()
        plt.suptitle(path, fontsize = 6)
        plt.savefig('all image.jpeg', dpi = 300)

    elif nImg == 4:
        fig, ax = plt.subplots(2,2, figsize = size_to_AN(6),dpi = 300)
        images_reshape = np.reshape(lstImg, (2,2)).T

        for i in range(2):
            for j in range(2):
                img = mpimg.imread(images_reshape[i][j])
                ax[i][j].imshow(img)
                ax[i][j].axis('off')

        plt.tight_layout()
        plt.suptitle(path, fontsize = 6)
        plt.savefig('all image.jpeg', dpi = 300)


    elif nImg == 5:
        fig, ax = plt.subplots(2,3, figsize = size_to_AN(6),dpi = 300)
        img0 = mpimg.imread(lstImg[0])
        ax[0][0].imshow(img0)
        ax[0][0].axis('off')

        img1 = mpimg.imread(lstImg[1])
        ax[0][1].imshow(img1)
        ax[0][1].axis('off')

        img2 = mpimg.imread(lstImg[2])
        ax[0][2].imshow(img2)
        ax[0][2].axis('off')

        img3 = mpimg.imread(lstImg[3])
        ax[1][0].imshow(img0)
        ax[1][0].axis('off')

        img4 = mpimg.imread(lstImg[4])
        ax[1][1].imshow(img1)
        ax[1][1].axis('off')

        ax[1][2].axis('off')

        plt.tight_layout()
        plt.suptitle(path, fontsize = 6)
        plt.savefig('all image.jpeg', dpi = 300)


    else:
        fig, ax = plt.subplots(2,3, figsize = size_to_AN(6),dpi = 300)
        images_reshape = np.reshape(lstImg, (2,2)).T

        for i in range(2):
            for j in range(3):
                img = mpimg.imread(images_reshape[i][j])
                ax[i][j].imshow(img)
                ax[i][j].axis('off')

    plt.tight_layout()
    plt.suptitle(path, fontsize = 6)
    plt.savefig('all image.jpeg', dpi = 300)

def SNR(xdata, ydata, fit_func, ini_guess, nsigma=4, factor=16*2/53, fname='default'):
    
    _popt, cov = scipy.optimize.curve_fit(fit_func, xdata, ydata, ini_guess) #fn, xdata, ydata, initial guess
    _popt = [_popt[0], _popt[1]*factor, _popt[2]*factor, _popt[3]]

    _max_val_ydata = max(ydata)
    _max_idx_ydata = ydata.tolist().index(_max_val_ydata)
    
    _noise_left = ydata[:int(_max_idx_ydata-nsigma*_popt[2])];
    _noise_right = ydata[int(_max_idx_ydata+nsigma*_popt[2]):];
    
    _noise = np.concatenate((_noise_left, _noise_right))
    
    _noise_baseline = np.mean(_noise)
    
    _noise_std = np.std(_noise)
    
    _shifted_max_val_ydata = _max_val_ydata - _noise_std
    
    SNR = 10*np.log10(_shifted_max_val_ydata/_noise_std)
    
    plt.figure(dpi=200)
    plt.plot(xdata*factor, ydata, '.', color='dodgerblue', alpha=0.9, label='data (Average Fluorescence)')
    plt.plot(xdata[:int(_max_idx_ydata-nsigma*_popt[2])]*factor, _noise_left, color='gray', alpha=0.8, label='background')
    plt.plot(xdata[int(_max_idx_ydata+nsigma*_popt[2]):]*factor, _noise_right, color='gray', alpha=0.8)
    plt.plot(xdata*factor, fit_func(xdata*factor, *_popt), 'orangered', alpha=0.8, label='fit')
    
    plt.axhline(y=_noise_baseline, linestyle='--', color='black', alpha=0.7, label='Average background')
    plt.text(0, _noise_baseline+5, 'Mean: '+ "{:.3f}".format(_noise_baseline), fontsize=5)

    plt.text(0, max(ydata)-30, 'Signal: '+ "{:.3f}".format(_max_val_ydata), fontsize=5)
    plt.text(0, max(ydata)-55, 'Std(Ave. bckgd): '+"{:.3f}".format(_noise_std), fontsize=5)
    plt.text(0, max(ydata)-90, 'SNR: '+"{:.3f}".format(SNR)+'dB', fontsize=5)
    plt.text(0, max(ydata)-120, 'Mean bckgd: '+"{:.3f}".format(_noise_baseline), fontsize=5)

    plt.text(0, max(ydata), fname, fontsize=5)
    plt.text(0, max(ydata)/2, 'Std(Gaussian): '+"{:.3f} um".format(_popt[2]*factor), fontsize=8)

    plt.xlabel(r'Size [$\mu$m]')
    plt.ylabel('Counts')
    plt.legend(fontsize='8', loc='upper right')
    
    plt.fill_between(xdata*factor, _noise_baseline+_noise_std, _noise_baseline-_noise_std, color='gray', alpha=0.35)
        
    return SNR, _noise

#this should be a method for ImportData
def scan2D(Analysis, runs, roi, scanVariables, calibrations=None, interpolation_type='none',calfunVar1=None,calfunVar2=None):
    
    x0 = roi['roi1'][0]
    y0 = roi['roi1'][1]
    lcrop = roi['roi1'][2]
    bgd_x0 = roi['bgd'][0]
    bgd_y0 = roi['bgd'][1]

    ScanVar1 = scanVariables[0]
    ScanVar2 = scanVariables[1]
    
    original_array = Analysis.lst_img['Abs']

    # Calculate the number of groups
    num_groups = runs[1] #runs_dFluore

    # Calculate the size of each group along axis 2
    group_size = original_array.shape[2] // num_groups

    # Initialize an empty list to store the sliced groups
    sliced_groups = []

    # Loop through and slice the array into groups
    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        sliced_group = original_array[:, :, start:end]
        sliced_groups.append(sliced_group)

    # Convert the list of sliced groups into a single NumPy array
    sliced_array = np.stack(sliced_groups, axis=2)

    sliced_array = np.swapaxes(sliced_array, 2, 3)


    _full_scan =[]
    for j in range(np.shape(sliced_array)[3]):
        _scan =[]
        for k in range(np.shape(sliced_array)[2]):
            image_k = sliced_array[:,:,k,j]
            image_k_crop = image_k[y0:y0+lcrop, x0:x0+lcrop] - image_k[bgd_y0:bgd_y0+lcrop, bgd_x0:bgd_x0+lcrop] #ROI - Bckgd

            _scan.append(np.sum(image_k_crop))
        _full_scan.append(_scan)
        
    if calibrations == 'MHz':
        xmin = calfunVar1(np.min(ScanVar1['Values']))
        xmax = calfunVar1(np.max(ScanVar1['Values']))
        ymin = calfunVar2(np.min(ScanVar2['Values']))
        ymax = calfunVar2(np.max(ScanVar2['Values']))
        unit = ' [MHz]'
    else:
        if calibrations == 'ms':
            unit = ' [ms]'
        else:
            unit= ''
        xmin = np.min(ScanVar1['Values'])
        xmax = np.max(ScanVar1['Values'])
        ymin = np.min(ScanVar2['Values'])
        ymax = np.max(ScanVar2['Values'])

    fig = plt.figure(dpi=200)
    im = plt.imshow(_full_scan, aspect='auto', 
                    cmap='plasma', origin='lower', 
                    interpolation=interpolation_type, 
                    extent=[xmin, xmax, ymin, ymax])
    cbar = fig.colorbar(im)
    cbar.set_clim(np.min(_full_scan), np.max(_full_scan))
    cbar.set_label('Fluorescence')
    plt.xlabel(ScanVar1['Name']+unit)
    plt.ylabel(ScanVar2['Name']+unit)

    return _full_scan

def convert_dFluoco(dFluoco,dD1Fluo,calibration_params=std_params):
    co = calibration_params['crossover'] - calibration_params['sp_after_crossover_D1'] + calibration_params['D1_spec']['a']*dD1Fluo + calibration_params['D1_spec']['b']\
    - (calibration_params['cool_double_3d']['a']*dFluoco + calibration_params['cool_double_3d']['b']) + calibration_params['cool_single_3d'] -10
    return co

def convert_dFluore(dFluore,dD1Fluo,calibration_params=std_params):
    re = -calibration_params['crossover'] - calibration_params['sp_after_crossover_D1'] + calibration_params['D1_spec']['a']*dD1Fluo + calibration_params['D1_spec']['b']\
    + (calibration_params['repumper_double']['a']*dFluore + calibration_params['repumper_double']['b']) + calibration_params['repumper_single']+10
    return re

def import_jpeg_images(folder_path):
    '''
    Import all JPEG images in a folder and return them as a list of PIL Image objects.

    Args:
    - folder_path: The path to the folder containing the JPEG images.

    Returns:
    - A list of PIL Image objects.
    '''
    image_list = []

    # Check if the folder path exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)

        # Filter JPEG files
        jpeg_files = [file for file in files if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg")]

        # Import JPEG images as PIL Image objects
        for jpeg_file in jpeg_files:
            image_path = os.path.join(folder_path, jpeg_file)
            try:
                image = Image.open(image_path)
                image_list.append(image)
            except Exception as e:
                print(f"Failed to open '{jpeg_file}': {e}")

    return image_list

#this should be a method for ImportData
def convolution2D(image, kernel):
    
    '''
        Convolution 2D
        
        Function that yields the convolution in 2 dimensions of an image given a kernel
        
        image (array): matrix representing the image to be convolve
        kernel (array): matrix representing the kernel to convolve the image image(*)kernel
    '''
    result = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

    return result

def convert_to_db(image):
    '''
        Convertion into Decibels (dB)
        
        Function that convert values of matrix that represent the image into values of logarithm 
        power (dB)
    '''
    
    image_db = 10 * np.log10(image)

    return image_db


#ANDOR CAMERA CONVERTION (maybe should go in the driver inself)

def ADC2photon(N_ADC, pixels, camera_params):
    '''
        Conversion analog-digital counts to photons
    '''
    tot_bias = pixels*camera_params['bias']; #compute total bias
    N_ph = np.abs(N_ADC - tot_bias)*camera_params['S']/(camera_params['QE']*camera_params['G']); #compute total number of photons
    return N_ph

def photon2ADC(N_ph, pixels, camera_params):
    '''
        Conversion photons to analog-digital counts
    '''

    tot_bias = pixels*camera_params['bias']; #compute total bias
    N_ADC = N_ph*camera_params['QE']*camera_params['G']/camera_params['S'] + tot_bias; #compute total number of counts
    return N_ADC

def Rayleigh_c(NA, wavelength):
    return 0.61*wavelength/NA
