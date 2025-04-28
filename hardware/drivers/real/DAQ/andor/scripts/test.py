from andor import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc

import time, os, sys

import cv2
from PIL import Image


TSTART = time.time_ns();

ixon897 = andor();
ixon897.Configure(args={'FanMode': 2, #0: full, 1: low, 2: off
                        'AcquisitionMode': 4, #1:single scan, #2:accumulate, 3: kinetics, 4: fast kinetics, 5: run till abort
                        'TriggerMode': 0, #0: internal, 1: external, 6: external start, 10: software trigger
                        'ReadMode': 4, #0: full vertical binning, 1:multi-track, 2: random track, 3: sinlge track, 4: image
                        'ExposureTime': 0.01784,
                        'NumberAccumulations': 1,
                        'NumberKinetics': 1,
                        'KineticCycleTime': 0.02460,
                        'VSSpeed': 4,
                        'VSAmplitude': 0,
                        'HSSpeed': [0,0],
                        'PreAmpGain': 2,
                        'ImageParams': {'hbin':1, 
                                        'vbin':1, 
                                        'hstart':1, 
                                        'hend':512, 
                                        'vstart':1,
                                        'vend':512}});

fig, ax = plt.subplots(figsize=(10,10))
fig.canvas.manager.set_window_title('Live image') 

ax.set_aspect(1.);

divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

binwidth = 0.25;
plotON = True;

TLOAD = time.time_ns();
print((TLOAD-TSTART)/1e9)

os.chdir(os.getcwd()+'\\images\\');

try:
	_counter=0;
	while True:

		TACQi = time.time_ns();

		image= [];
		ixon897.StartAcquisition();
		# time.sleep(0.4) ;

		ixon897.GetAcquiredData(image);
		image= np.array(image);
		image= image.reshape((ixon897.width, ixon897.height));
		
		TACQf = time.time_ns();
		print((TACQf-TACQi)/1e9)

		# im = Image.fromarray(image)
		# im.save(f'outfile_'+str(_counter)+'.png', 'PNG')

		if plotON == True:
			if _counter==0:
				im = ax.imshow(image, cmap='gray')
				ax.set_title('Raw image');
				cbar = plt.colorbar(im);

				x = image[:][256];
				y = image[256][:];

				xymax = max(np.max(np.abs(x)), np.max(np.abs(y)));
				lim = (int(xymax/binwidth) + 1)*binwidth;
				bins = np.arange(-lim, lim + binwidth, binwidth);

				ax_histx.hist(x, bins=bins);
				ax_histy.hist(y, bins=bins, orientation='horizontal');

				ax_histx.set_yticks([0, 200, 400, 512]);
				ax_histy.set_xticks([0, 200, 400, 512]);

				ax.set_xlim(0,512);
				ax.set_ylim(0,512);

				plt.show(block=False);


			else:
				im.set_data(image);
				plt.pause(0.2);

		_counter+=1;

except KeyboardInterrupt:
    	    
	plt.show();

	ixon897.ShutDown();



def create_video(image_folder, video_name, fps=25):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# # Provide the path to the folder containing the images
# image_folder = os.getcwd()

# # Specify the output video name and frame rate (default is 25 fps)
# video_name = "output.mp4"
# fps = 30

# # Call the function to create the video
# create_video(image_folder, video_name, fps)