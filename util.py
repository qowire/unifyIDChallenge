import plotly.plotly as py
import numpy as np
import plotly.graph_objs as go
import pickle

'''
Citations:
[1] Stan Salvador, and Philip Chan. “FastDTW: Toward accurate dynamic time warping in linear time and space.” Intelligent Data Analysis 11.5 (2007): 561-580.
[2] Derawi, Mohammad O., Patrick Bours, and Kjetil Holien. "Improved cycle detection for accelerometer based gait authentication." 2010 Sixth International Conference on Intelligent Information Hiding and Multimedia Signal Processing. IEEE, 2010.
[3] Rong, Liu, et al. "Identification of individual walking patterns using gait acceleration." Bioinformatics and Biomedical Engineering, 2007. ICBBE 2007. The 1st International Conference on. IEEE, 2007.
[4] Rong, Liu, et al. "A wearable acceleration sensor system for gait recognition." Industrial Electronics and Applications, 2007. ICIEA 2007. 2nd IEEE Conference on. IEEE, 2007.
[5] Keogh, Eamonn, and Chotirat Ann Ratanamahatana. "Exact indexing of dynamic time warping." Knowledge and information systems 7.3 (2005): 358-386.
[6] Stamatakis, Julien, et al. "Gait feature extraction in Parkinson's disease using low-cost accelerometers." Engineering in Medicine and Biology Society, EMBC, 2011 Annual International Conference of the IEEE. IEEE, 2011.
[7] Selles, Ruud W., et al. "Automated estimation of initial and terminal contact timing using accelerometers; development and validation in transtibial amputees and controls." IEEE transactions on neural systems and rehabilitation engineering 13.1 (2005): 81-88.
[8] Rueterbories, Jan, et al. "Methods for gait event detection and analysis in ambulatory systems." Medical engineering & physics 32.6 (2010): 545-552.

'''

NUM_CYCLES_TO_PLOT = 20
INTERPOLATION_FREQUENCY = 100 #Hz
NUM_AXES = 3


def plot_curves(curves, labels):
	x = np.arange(len(curves[0]))
	traces = []
	for i in range(len(curves)):
		trace = go.Scatter(
			x=x,
			y=curves[i],
			name=labels[i])
		traces.append(trace)
	layout = go.Layout(
		title='Example Curves')
	fig = go.Figure(data=traces, layout=layout)
	py.plot(fig)

def plot_losses(train_losses, dev_losses):
	x = np.arange(len(train_losses))
	train_trace = go.Scatter(
		x=x,
		y=train_losses,
		name='Training Loss'
		)
	dev_trace = go.Scatter(
		x=x,
		y=dev_losses,
		name='Dev Loss'
		)
	data = [train_trace, dev_trace]
	layout = go.Layout(
	    title='Train Loss vs Dev Loss',
	    yaxis=dict(
	        title='MSE (L2) Loss'
    	),
    	xaxis=dict(
    		title='Iterations'
		)
	)
	fig = go.Figure(data=data, layout=layout)
	py.plot(data)


def plot_cycles(cycles):
	data = []
	if len(cycles) > NUM_CYCLES_TO_PLOT:
		indices = np.random.choice(len(cycles), NUM_CYCLES_TO_PLOT, True)
		cycles = [cycles[index] for index in indices]
	x = np.arange(GAIT_LENGTH)
	for i, cycle in enumerate(cycles):
		trace = go.Scatter(
			x=x,
			y=cycle,
			name='Step' + str(i)
			)
		data.append(trace)
	layout = go.Layout(
	    title='Overlay of random Sample of 10 steps of wx',
	    yaxis1=dict(
	        title='Normalized wx'
    	),
    	xaxis1=dict(
    		title='Time (s)'
		)
	)
	fig = go.Figure(data=data, layout=layout)
	py.plot(data)

def plot_full_data(data, labels):
	x_shape, axis, dtype = data.shape
	x = list(range(x_shape))
	traces = []
	for a in range(axis):
		for d in range(dtype):
			trace = go.Scatter(
				x = x,
				y = data[:,a,d],
				name = labels[a][d],
				yaxis = 'y' + str(a + 1)
			)
			traces.append(trace)
	layout = go.Layout(
	    title='10 steps Correlation between Accelerometer and Gyroscope',
	    yaxis1=dict(
	        title='Accelerometer Readings'
    	),
    	yaxis2=dict(
	        title='Gyroscope Readings',
	        titlefont=dict(
	            color='rgb(148, 103, 189)'
	        ),
	        tickfont=dict(
	            color='rgb(148, 103, 189)'
	        ),
	        overlaying='y',
	        side='right'
    	),
    	xaxis1=dict(
    		title='Time (s)'
		)
	)
	fig = go.Figure(data=traces, layout=layout)
	_ = py.plot(fig);


# Calculates linear interpolation of data to have standardized
# indices equal to t = index / INTERPOLATION_FREQUENCY
# Inputs: pd dataframe with ['time'] column
# Returns: 3D np array corresponding to
#	(time) x (accel, gyro) x (xyz)
# 
# If machine learning model is not performing well,
# 	can do more signal processing here. Papers seem
# 	to generally propose wavelet decomposition, more
#	specifically Daubechies wavelet of order 8 [3, 4]
# 	and 4th order Butterworth filters [6, 7, 8]. 
#  	However, more signal processing did not seem to be
# 	necessary in this case, as the wx gyroscope data
#  	was not too noisy and relatively easy to interpret. 
def preprocess_data(accel_data, gyro_data):
	accel_data['time'] -= accel_data['time'][0]
	gyro_data['time'] -= gyro_data['time'][0]
	num_seconds = int(min(accel_data.time.iloc[-1],
						gyro_data.time.iloc[-1]))
	num_elements = INTERPOLATION_FREQUENCY * num_seconds
	# k x (accel, gyro) x (xyz) 
	preprocessed_data = np.zeros((num_elements, 2, NUM_AXES)) 
	x = np.arange(0, num_seconds, 1. / INTERPOLATION_FREQUENCY)
	for k in range(len(preprocessed_data[0][0])):
		preprocessed_data[:,0,k] = np.interp(
			x, list(accel_data.iloc[:,0]), list(accel_data.iloc[:,k+1]))
		preprocessed_data[:,1,k] = np.interp(
			x, list(gyro_data.iloc[:,0]), list(gyro_data.iloc[:,k+1]))
	preprocessed_data[:,0,:] -= np.mean(preprocessed_data[:,0,:])
	preprocessed_data[:,0,:] /= np.amax(np.abs(preprocessed_data[:,0,:]))
	preprocessed_data[:,1,:] -= np.mean(preprocessed_data[:,1,:])
	preprocessed_data[:,1,:] /= np.amax(np.abs(preprocessed_data[:,1,:]))
	return preprocessed_data


def time_convert(x):
	try:
		m, s = map(float, x.split(':'))
		return 60 * m + s
	except Exception as e:
		d, t = x.split(' ')
		h, m, s = map(float, t.split(':'))
		return 3600 * h + 60 * m + s

def load_pickle(file_name):
	infile = open(file_name, 'rb')
	pickled_object = pickle.load(infile)
	infile.close()
	return pickled_object

def dump_pickle(unpickled_object, file_name):
	outfile = open(file_name, 'wb')
	pickle.dump(unpickled_object, outfile)
	outfile.close()