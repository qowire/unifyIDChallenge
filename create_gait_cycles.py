import numpy as np
import pandas as pd
from fastdtw import fastdtw
import os.path
import util as U

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


ACCEL_SMALL = './data/accel_small.csv'
GYRO_SMALL = './data/gyro_small.csv'
ACCEL_SMALL_CACHE = '__pycache__/accel_small.npy'
GYRO_SMALL_CACHE = '__pycache__/gyro_small.npy'
MVMT_SMALL_CACHE = '__pycache__/mvmt_small.npy'
CYCLE_SMALL_CACHE = '__pycache__/cycle_small.npy'

ACCEL_LARGE = './data/gyroscope_front_large.csv'
GYRO_LARGE = './data/gyroscope_back_large.csv'
ACCEL_LARGE_CACHE = '__pycache__/accel_large.npy'
GYRO_LARGE_CACHE = '__pycache__/gyro_large.npy'
MVMT_LARGE_CACHE = '__pycache__/mvmt_large.npy'
CYCLE_LARGE_CACHE = '__pycache__/cycle_large.npy'

GAIT_LENGTH = 100

OUTLIER_STD_TOLERANCE = 2

USE_SMALL = True
CREATE_PLOT = True

# Calculates the cycle cutoffs of the input
# Uses the wx readings, which seem to be the most regular
# Graphs of all 6 readings can be seen in 
# ./plots/gyroscope_accelerometer_relationship.png
# Computes a cycle based on observation that cycles begin regularly
# with wx being positive for a large amount of time
# 
# Input: 1D numpy array, corresponding to 
# 	(time) x (x) of the gyroscope data
# Returns: 1D numpy array, corresponding to
# 	start and end points. Has N + 1 points for N cycles. 

def separate_gait_cycles(wx):
	upward_cross_points = []
	lengths_of_positive_segments = []
	cur_index = 0
	# move cur_index to beginning of gait cycle where gait is under curve
	while wx[cur_index] >= 0:
		cur_index += 1
	# since we begin with the gait under the curve, upward cross points will always be 
	# equal to or longer than downward cross points
	while cur_index < len(wx) - 1:
		if wx[cur_index] <= 0 and wx[cur_index + 1] > 0:
			upward_cross_points.append(cur_index)
		elif wx[cur_index] > 0 and wx[cur_index + 1] <= 0:
			lengths_of_positive_segments.append(cur_index - upward_cross_points[-1])
		cur_index += 1
	# use length of downward cross points to be conservative in array indexing
	mean_length = np.mean(lengths_of_positive_segments)
	upward_cross_points = np.array(upward_cross_points)
	gait_points = np.append(upward_cross_points[lengths_of_positive_segments > mean_length], len(wx) - 1)
	return gait_points

# Computes distance between cycles using FastDTW[1] and removes
# cycles which have an abnormally high average distance to other cycles. 
# Inspired by [2], and usage of DTW to compute these distances seems
# to be common practice [3, 4, 5]. 
# 
# Input: List of 1D numpy arrays, corresponding a list of cycles reduced to the wx signals
# Returns: List of 1D numpy arrays, corresponding to the original list with outliers
# 	removed

def remove_outlier_cycles(cycles):
	dist_matrix = np.zeros((len(cycles), len(cycles)))
	for i, cycle in enumerate(cycles):
		for j in range(i + 1, len(cycles)):
			dist_matrix[i][j], _ = fastdtw(cycles[i], cycles[j])
			dist_matrix[j][i] = dist_matrix[i][j]
	avg_dist_vec = np.zeros(len(cycles))
	for i in range(len(cycles)):
		avg_dist_vec[i] = np.sum(dist_matrix[i]) * 1. / (len(cycles) - 1)
	mean, std = np.mean(avg_dist_vec), np.std(avg_dist_vec)
	lower_bound = avg_dist_vec > mean - OUTLIER_STD_TOLERANCE * std
	upper_bound = avg_dist_vec < mean + OUTLIER_STD_TOLERANCE * std
	return [cycles[i] for i in range(len(lower_bound)) if lower_bound[i] and upper_bound[i]]

# Normalizes cycle lenghths using linear interpolation to have standard input
# to a ML model. We can also use unequal input lengths if we use a different ML 
# model, such as an RNN

# Input: list of 1D numpy arrays, corresponding to a list of cycles produced
#  from remove_outlier_cycles
# Operates in place and does not return anything. 

def normalize_cycles(cycles):
	x = np.arange(GAIT_LENGTH)
	for i in range(len(cycles)):
		cycles[i] = np.interp(x, np.arange(len(cycles[i])), cycles[i])

def main():
	if USE_SMALL:
		accel_file, gyro_file = ACCEL_SMALL, GYRO_SMALL
		mvmt_cache, normalized_cycle_cache = MVMT_SMALL_CACHE, CYCLE_SMALL_CACHE
	else:
		accel_file, gyro_file = ACCEL_LARGE, GYRO_LARGE
		mvmt_cache, normalized_cycle_cache = MVMT_LARGE_CACHE, CYCLE_LARGE_CACHE

	accel_csv, gyro_csv = pd.read_csv(accel_file), pd.read_csv(gyro_file)
	if accel_csv['time'].dtype != float:
		accel_csv['time'] = accel_csv.time.apply(U.time_convert)
	if gyro_csv['time'].dtype != float:
		gyro_csv['time'] = gyro_csv.time.apply(U.time_convert)

	try:
		mvmt_data = U.load_pickle(mvmt_cache)
		print('mvmt_data loaded from pickle.')
	except Exception as e:
		print('mvmt_data unable to be loaded from pickle. Generating mvmt_data...')
		mvmt_data = U.preprocess_data(accel_csv, gyro_csv)
		U.dump_pickle(mvmt_data, mvmt_cache)
	wx_data = mvmt_data[:,1,0]

	try:
		gait_cycles = U.load_pickle(normalized_cycle_cache)
		print('gait_cycles loaded from pickle.')
	except Exception as e:
		print('gait_cycles unable to be loaded from pickle. Generating gait_cycles...')
		print('Separating cycles...')
		gait_cycle_points = separate_gait_cycles(wx_data)
		gait_cycles = [wx_data[gait_cycle_points[i]:gait_cycle_points[i+1]] for i in range(len(gait_cycle_points) - 1)]
		print('Removing outliers...')
		gait_cycles = remove_outlier_cycles(gait_cycles)
		print('Normalizing cycles...')
		normalize_cycles(gait_cycles)
		U.dump_pickle(gait_cycles, normalized_cycle_cache)

	if CREATE_PLOT:
		print('Plotting data')
		mvmt_labels = [['ax', 'ay', 'az'], ['wx', 'wy', 'wz']]
		U.plot_full_data(mvmt_data, mvmt_labels)
		U.plot_cycles(gait_cycles)

if __name__ == '__main__':
	main()