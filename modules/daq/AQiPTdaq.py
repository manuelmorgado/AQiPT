#Atomic Quantum information Processing Tool (AQIPT) - DAQ module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2022-04-11
# Last update: 2023-05-30

import os, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
from PIL import Image

import plotly.graph_objects as go
# import plotly.graph_objs as go
# import plotly.express as px


from AQiPT import AQiPTcore as aqipt
# from AQiPT.modules.analysis import AQiPTanalysis as analysis
from AQiPT.hardware.drivers.real.DAQ.andor.andor import *
# from AQiPT.hardware.drivers.AQiPTrd import drivers
# import AQiPT.hardware.drivers.AQiPTvd as vdrivers

from AQiPT.modules.directory import AQiPTdirectory as dirPath

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Output, Input
from flask import Flask


SEQUENCE_PORT = 8059;
CAMERA_PORT = 8058;

#####################################################################################################
#Inspector AQiPT class
#####################################################################################################
class inspector:

    def __init__(self, name, kind, chnls_map, LANid, slave=True, statusMode=False):

        #atributes
        self._dashboard = None;
        self._datamanager = None;



class dashboard:

    def __init__(self):

        self.server = Flask(__name__);

    def add_dash4Sequences(self, sequence_lst):
        plotSequences(sequence_lst=sequence_lst, 
                      dash_params={'debugger': True,
                                   'reloader': False,
                                   'url_base_pathname':'/'+'Sequence'+'/',
                                   'server': self.server,
                                   'port': 8050,
                                   'name': 'Waveforms'
                                  },
                       run=False
                      );

    def add_dash4LiveImage(self):
        pass

    def open(self):
        self.server.run_server();

class plotSequences(dashboard):

    def __init__(self, 
                 sequence_lst:list,
                 dash_params={'debugger': True,
                              'reloader': False,
                              'url_base_pathname':None,
                              'server': Flask(__name__),
                              'port': SEQUENCE_PORT,
                              'name': 'Waveforms'},
                  run=True):


        self.plot_lst = sequence_lst;
        self.__graphs_lst = [dcc.Graph(figure=PLOT) for PLOT in self.plot_lst];
        self._dash_params = dash_params;
        self.app = dash.Dash(__name__, server=self._dash_params['server'], url_base_pathname=self._dash_params['url_base_pathname']);


        self.app.layout = html.Div(self.__graphs_lst);
        self.app.title = self._dash_params['name'];

        if run==True:
            self.app.run_server(debug = self._dash_params['debugger'], 
                                use_reloader = self._dash_params['reloader'], 
                                port = self._dash_params['port']);  # Turn off reloader if inside Jupyter

class plotLiveimage(dashboard):

    def __init__(self, 
                 camera,
                 camera_name= 'ixon897',
                 camera_params={'FanMode': 2, #0: full, 1: low, 2: off
                                'AcquisitionMode': 3, #1:single scan, #2:accumulate, 3: kinetics, 4: fast kinetics, 5: run till abort
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
                                                'vend':512}},
                 plotON=True, 
                 plot_params={'figure_size': (10,10),
                              'binwidth': 0.25,
                              'colormap': 'gray',
                              'figure_title': 'Raw image'
                             },
                 saveImages=False,
                 images_params={'path2imagesfolder': '\\images',
                                'image_format': 'PNG',
                                'image_label': 'outfile'},
                 dash_params={'debugger': True,
                              'reloader': False,
                              'url_base_pathname':None,
                              'server': Flask(__name__),
                              'port': CAMERA_PORT,
                              'name': 'Live image'},
                 run=True):

        self.camera = camera;
        self.camera_name = camera_name;
        self.camera_params = camera_params;
        self.plotON = plotON;
        self.plot_params = plot_params;
        self.saveImages = saveImages;
        self._dash_params = dash_params;
        self.images_params = images_params;
        self.app = dash.Dash(__name__);

        self.app.layout = html.Div([html.H1("Live Camera Image"),
                                    dcc.Graph(id='live-image'),
                                    dcc.Interval(id='interval-component',
                                                 interval=800,  # Update the dashboard every second
                                                 n_intervals=0)
                                  ]);
        self.app.title = self._dash_params['name'];
        self.app.callback( [Output('live-image', 'figure')], [Input('interval-component', 'n_intervals')]);

        if run==True:
            self.app.run_server(debug = self._dash_params['debugger'], 
                                use_reloader = self._dash_params['reloader'], 
                                port = self._dash_params['port']);  # Turn off reloader if inside Jupyter

    def _get_frame(self):
        
        if self.camera_name == 'ixon897':
            self.camera.Configure(args=camera_params);

        while True:
            image= [];
            self.camera.StartAcquisition();
            time.sleep(0.4) ;

            self.camera.GetAcquiredData(image);
            image= np.array(image);
            image= image.reshape((self.camera.width, self.camera.height));

            yield image


    def update_graphs(n):
        # Get the latest frames from the webcam
        frames = _get_frame()
        frame = next(frames)
        
        # Update the image
        fig_image = px.imshow(frame, binary_format="jpeg", binary_compression_level=0)
        
        return fig_image




def plotLiveCamera(camera,
                   camera_name= 'ixon897',
                   camera_params={'FanMode': 2, #0: full, 1: low, 2: off
                                  'AcquisitionMode': 3, #1:single scan, #2:accumulate, 3: kinetics, 4: fast kinetics, 5: run till abort
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
                                                  'vend':512}},
                    plotON=True, 
                    plot_params={'figure_size': (10,10),
                                 'binwidth': 0.25,
                                 'colormap': 'gray',
                                 'figure_title': 'Raw image'
                                },
                    saveImages=False,
                    images_params={'path2imagesfolder': '\\images',
                                   'image_format': 'PNG',
                                   'image_label': 'outfile'}):

    if camera_name == 'ixon897':
        ixon897 = camera;
        ixon897.Configure(args=camera_params);

        fig, ax = plt.subplots(figsize=plot_params['figure_size']);
        fig.canvas.manager.set_window_title('Live image') ;
        ax.set_aspect(1.);

        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax);
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax);

        if saveImages== True:
            os.chdir(images_params['path2imagesfolder']);

        try:
            _counter=0;
            while True:


                image= [];
                ixon897.StartAcquisition();
                time.sleep(0.4) ;

                ixon897.GetAcquiredData(image);
                image= np.array(image);
                image= image.reshape((ixon897.width, ixon897.height));
                
                if saveImages== True:
                    im = Image.fromarray(image);
                    if images_params['image_format'] == 'PNG':
                        im.save(images_params['image_label']+'_'+str(_counter)+'.png', 'PNG');

                if plotON == True:
                    if _counter==0:
                        im = ax.imshow(image, cmap=plot_params['colormap']);
                        ax.set_title(plot_params['figure_title']);
                        cbar = plt.colorbar(im);

                        x = image[:][int(ixon897.width/2)];
                        y = image[int(ixon897.height/2)][:];

                        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)));
                        lim = (int(xymax/plot_params['binwidth']) + 1)*plot_params['binwidth'];
                        bins = np.arange(-lim, lim + plot_params['binwidth'], plot_params['binwidth']);

                        ax_histx.hist(x, bins=bins);
                        ax_histy.hist(y, bins=bins, orientation='horizontal');

                        ax_histx.set_yticks([0, 200, 400, ixon897.width/2]);
                        ax_histy.set_xticks([0, 200, 400, ixon897.width/2]);

                        ax.set_xlim(0,ixon897.width/2);
                        ax.set_ylim(0,ixon897.width/2);

                        plt.show(block=False);

                        _counter+=1;

                    else:
                        im.set_data(image);
                        plt.pause(0.4);

        except KeyboardInterrupt:
                    
            plt.show();

            ixon897.ShutDown();
        self.app.run_server(debug=True, use_reloader=False, port=8051)



def create_video(image_folder, video_name= "output.mp4", fps=25):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()




class graph:

    def __init__(self, *args, **kwargs):

        self._datasets = [kwargs['data']];
        self._processedData = None;
        self.plots = [];

    def createSurface3D(self, data2plot=None):

        if data2plot is None:
            _data = self._datasets
        else:
            _data = data2plot;

        _s3D_plot = surface3D(data=_data);
        _s3D_plot.plot();
        self.plots.append(_s3D_plot);

    def colormap2D(self, data=None):

        if data is None:
            _data = self._datasets
        else:
            _data = data;
            
        _cm2D_plot = self.colormap2D(data=_data);
        _cm2D_plot.plot();
        self.plots.append(_cm2D_plot);

    def createPlot1D(self, data=None):

        if data is None:
            _data = self._datasets
        else:
            _data = data;
            
        _p1D_plot = self.plot1D(data=_data);
        _p1D_plot.plot();
        self.plots.append(_p1D_plot);

    def scatter2D(self, data=None):

        if data is None:
            _data = self._datasets
        else:
            _data = data;
            
        _sc2D_plot = self.scatter2D(data=_data);
        _sc2D_plot.plot();
        self.plots.append(_sc2D_plot);

    def scatter3D(self, data=None):

        if data is None:
            _data = self._datasets
        else:
            _data = data;
            
        _sc3D_plot = self.scatter3D(data=_data);
        _sc3D_plot.plot();
        self.plots.append(_sc3D_plot);

class surface3D(graph):

    def __init__(self, *args, **kwargs):
        
        self._frames = []
        self._rawData = [kwargs['data']];
        self._dimensions = None;
        self._processedData = None;
        self.plots = None;
        self.figure = None;
        
        if isinstance(kwargs['data'], list):
            self._frames += [pd.DataFrame(dataset) for dataset in kwargs['data']];

        

    def plot(self):

        if len(self._frames)>1:
            df = self._frames
        else:
            self._rawData = pd.concat(self._frames);
            # df = pd.DataFrame(self._rawData);
            df = self._rawData;

        # create figure
        fig = go.Figure()

        # Add surface trace
        for _ddf in df:
            _ddf = pd.DataFrame(_ddf);

            fig.add_trace(go.Surface(z=_ddf.values.tolist(), colorscale="Viridis"))

        # Update plot sizing
        fig.update_layout(
            width=800,
            height=900,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )

        # Update 3D scene options
        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=["type", "surface"],
                            label="3D Surface",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "heatmap"],
                            label="Heatmap",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(text="Trace type:", showarrow=False,
                x=0, y=1.085, yref="paper", align="left")
            ]
        )

        self.figure = fig;
        fig.show()

    def includeData(self, newData):
        if not isinstance(newData, pd.DataFrame):
            self._frames.append(newData);
            df = pd.DataFrame(newData);

        self.figure.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))
        self.figure.show()

class colormap2D(graph):
    pass

class plot1D(graph):

    def __init__(self, *args, **kwargs):
        
        self._Xframes = [];
        self._Yframes = [];
        self._XrawData = [kwargs['x_data']];
        self._YrawData = [kwargs['y_data']];
        self._dimensions = None;
        self._processedData = None;
        self.plots = None;
        self.figure = None;
        
        if isinstance(kwargs['x_data'], list):
            self._Xframes += [pd.DataFrame(dataset) for dataset in kwargs['x_data']];

        if isinstance(kwargs['y_data'], list):
            self._Yframes += [pd.DataFrame(dataset) for dataset in kwargs['y_data']];

        
    def plot(self):

        if len(self._Xframes)>1 and len(self._Yframes)>1:
            xdf = self._Xframes;
            ydf = self._Yframes;
        else:
            self._XrawData = pd.concat(self._Xframes);
            # df = pd.DataFrame(self._rawData);
            xdf = self._XrawData;

            self._YrawData = pd.concat(self._Yframes);
            # df = pd.DataFrame(self._rawData);
            ydf = self._YrawData;

        # Initialize figure
        fig = go.Figure()

        # Add Trace
        for _xddf,_yddf in zip(xdf,ydf):
            _xddf = pd.DataFrame(_xddf);
            _yddf = pd.DataFrame(_yddf);


            fig.add_trace(go.Scatter(x=_xddf.values.tolist(),
                                     y=_yddf.values.tolist(),
                                     name="High",
                                     line=dict(color="#33CFA5")))


        # Add Annotations and Buttons
        # high_annotations = [dict(x="2016-03-01",
        #                          y=df.High.mean(),
        #                          xref="x", yref="y",
        #                          text="High Average:<br> %.3f" % df.High.mean(),
        #                          ax=0, ay=-40),
        #                     dict(x=df.Date[df.High.idxmax()],
        #                          y=df.High.max(),
        #                          xref="x", yref="y",
        #                          text="High Max:<br> %.3f" % df.High.max(),
        #                          ax=-40, ay=-40)]
        # low_annotations = [dict(x="2015-05-01",
        #                         y=df.Low.mean(),
        #                         xref="x", yref="y",
        #                         text="Low Average:<br> %.3f" % df.Low.mean(),
        #                         ax=0, ay=40),
        #                    dict(x=df.Date[df.High.idxmin()],
        #                         y=df.Low.min(),
        #                         xref="x", yref="y",
        #                         text="Low Min:<br> %.3f" % df.Low.min(),
        #                         ax=0, ay=40)]

        # fig.update_layout(
        #     updatemenus=[
        #         dict(
        #             active=0,
        #             buttons=list([
        #                 dict(label="None",
        #                      method="update",
        #                      args=[{"visible": [True, False, True, False]},
        #                            {"title": "Yahoo",
        #                             "annotations": []}]),
        #                 dict(label="High",
        #                      method="update",
        #                      args=[{"visible": [True, True, False, False]},
        #                            {"title": "Yahoo High",
        #                             "annotations": high_annotations}]),
        #                 dict(label="Low",
        #                      method="update",
        #                      args=[{"visible": [False, False, True, True]},
        #                            {"title": "Yahoo Low",
        #                             "annotations": low_annotations}]),
        #                 dict(label="Both",
        #                      method="update",
        #                      args=[{"visible": [True, True, True, True]},
        #                            {"title": "Yahoo",
        #                             "annotations": high_annotations + low_annotations}]),
        #             ]),
        #         )
        #     ])

        # Set title
        fig.update_layout(title_text="Yahoo")

        fig.show()

        self.figure = fig;
        fig.show()

    def includeData(self, newData):
        if not isinstance(newData, pd.DataFrame):
            self._frames.append(newData);
            df = pd.DataFrame(newData);

        self.figure.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))
        self.figure.show()

class scatter2D(graph):
    pass

class scatter3D(graph):
    pass

