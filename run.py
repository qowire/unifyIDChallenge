from datetime import datetime
import create_gait_cycles
from model import GaitModel
import pandas as pd
import torch
from torch import nn, optim
import math
import time
import random
import numpy as np
import os
import sys
import util as U

GYRO_FRONT = './data/gyroscope_front_large.csv'
GYRO_BACK = './data/gyroscope_back_large.csv'

DATA_CACHE = './__pycache__/training_data_cache.npy'

FRONT_TO_BACK_MODEL = False

HIDDEN_SIZE = 100
INPUT_SIZE = 100
OUTPUT_SIZE = 100
DROPOUT_PROB = 0.5

TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1

NUM_EPOCHS = 80


def read_data(file1, file2):
	file1_csv = pd.read_csv(file1)
	file2_csv = pd.read_csv(file2)
	if file1_csv['time'].dtype != float:
		file1_csv['time'] = file1_csv.time.apply(U.time_convert)
	if file2_csv['time'].dtype != float:
		file2_csv['time'] = file2_csv.time.apply(U.time_convert)
	stacked_data = U.preprocess_data(file1_csv, file2_csv)

	file1_data = stacked_data[:,0,0]
	file2_data = stacked_data[:,1,0]

	file1_cycles = create_gait_cycles.separate_gait_cycles(file1_data)
	file2_cycles = create_gait_cycles.separate_gait_cycles(file2_data)

	file1_gait_cycles = [file1_data[file1_cycles[i]:file1_cycles[i+1]] for i in range(len(file1_cycles) - 1)]
	file2_gait_cycles = [file2_data[file2_cycles[i]:file2_cycles[i+1]] for i in range(len(file2_cycles) - 1)]

	print('Cycles found. Removing outliers...')

	file1_gait_cycles = create_gait_cycles.remove_outlier_cycles(file1_gait_cycles)
	file2_gait_cycles = create_gait_cycles.remove_outlier_cycles(file2_gait_cycles)

	print('Outliers removed. Normalizing cycles...')

	create_gait_cycles.normalize_cycles(file1_gait_cycles)
	create_gait_cycles.normalize_cycles(file2_gait_cycles)

	return [file1_gait_cycles, file2_gait_cycles]

def train(model, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
	best_dev_loss = sys.maxsize
	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_func = nn.MSELoss()

	train_losses = []
	dev_losses = []
	for epoch in range(n_epochs):
		avg_train_loss, dev_loss = train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size)
		train_losses.append(avg_train_loss.item())
		dev_losses.append(dev_loss.item())
		if dev_loss < best_dev_loss:
			best_dev_loss = dev_loss
			torch.save(model.state_dict(), output_path)
	U.plot_losses(train_losses, dev_losses)

def minibatches(data, batch_size):
	return [data[:,i * batch_size:(i + 1) * batch_size,:] for i in range(math.ceil(len(data) / batch_size))]

def train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size):
	model.train()
	n_minibatches = math.ceil(len(train_data) / batch_size)
	total_loss = 0.
	num_updates = 0

	for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
		optimizer.zero_grad()
		loss = 0.
		train_x = torch.from_numpy(train_x).float()
		train_y = torch.from_numpy(train_y).float()
		logits = model.forward(train_x)
		loss = loss_func(logits, train_y)
		loss.backward()
		optimizer.step()

		total_loss += loss
		num_updates += 1

	model.eval()
	dev_x = torch.from_numpy(dev_data[0]).float()
	dev_y = torch.from_numpy(dev_data[1]).float()
	dev_loss = loss_func(model.forward(dev_x), dev_y)
	return total_loss / num_updates, dev_loss

def permute_and_split_data(data):
	front, back = data[0], data[1]
	random.shuffle(front)
	random.shuffle(back)

	split_front1 = int(len(front) * TRAIN_SPLIT)
	split_front2 = int(len(front) * (TRAIN_SPLIT + DEV_SPLIT))
	front_train = front[:split_front1]
	front_dev = front[split_front1:split_front2]
	front_test = front[split_front2:]

	split_back1 = int(len(back) * TRAIN_SPLIT)
	split_back2 = int(len(back) * (TRAIN_SPLIT + DEV_SPLIT))
	back_train = back[:split_back1]
	back_dev = back[split_back1:split_back2]
	back_test = back[split_back2:]

	front_back_lengths = [
		(split_front1, split_back1),
		(split_front2 - split_front1, split_back2 - split_back1),
		(len(front) - split_front2, len(back) - split_back2)
	]
	output_data = []
	x = [front_train, front_dev, front_test]
	y = [back_train, back_dev, back_test]
	for i, (front_len, back_len) in enumerate(front_back_lengths):
		data_len = front_len * back_len
		output_data.append(np.zeros((data_len, 2, INPUT_SIZE)))
		for j in range(front_len):
			for k in range(back_len):
				output_data[i][j * back_len + k][0] = x[i][j]
				output_data[i][j * back_len + k][1] = y[i][k]
		random.shuffle(output_data[i])
		output_data[i] = output_data[i].transpose(1, 0, 2)

	return output_data


def main():
	print(80 * "=")
	print("INITIALIZING")
	print(80 * "=")
	start = time.time()

	try:
		data = U.load_pickle(DATA_CACHE)
		print('data loaded from pickle.')
	except Exception as e:
		print('data unable to be loaded from pickle. Generating data...')
		data = read_data(GYRO_FRONT, GYRO_BACK)
		U.dump_pickle(data, DATA_CACHE)

	assert(torch.__version__ == "1.0.0"),  "Please install torch version 1.0.0"

	if not FRONT_TO_BACK_MODEL:
		data[0], data[1] = data[1], data[0]
	train_data, dev_data, test_data = permute_and_split_data(data)
	model = GaitModel(HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, DROPOUT_PROB)


	print("took {:.2f} seconds\n".format(time.time() - start))

	print(80 * "=")
	print("TRAINING")
	print(80 * "=")
	output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
	output_path = output_dir + "model.weights"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	train(model, train_data, dev_data, output_path, batch_size=1024, n_epochs=NUM_EPOCHS, lr=0.0005)

	print(80 * "=")
	print("TESTING")
	print(80 * "=")
	print("Restoring the best model weights found on the dev set")
	model.load_state_dict(torch.load(output_path))
	print("Final evaluation on test set",)
	model.eval()

	test_x = torch.from_numpy(test_data[0]).float()
	test_y = torch.from_numpy(test_data[1]).float()
	forwarded = model.forward(test_x)
	loss = nn.MSELoss()(forwarded, test_y)
	U.plot_curves(
		[forwarded[0].detach().numpy(), test_x[0].detach().numpy(), test_y[0].detach().numpy()],
		['Output curve', 'Original curve', '(Example) Reference Curve'])
	print("- test loss: {}".format(loss))
	print("Done!")



if __name__ == '__main__':
	main()