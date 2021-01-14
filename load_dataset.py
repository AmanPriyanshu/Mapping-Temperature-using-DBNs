import pandas as pd
import numpy as np
import torch

def min_max_x(x):
	for index, col in enumerate(x.T):
		min_col = np.min(col)
		max_col = np.max(col)
		if min_col != max_col:
			x.T[index] = (x.T[index] - min_col)/(max_col - min_col)
		else:
			x.T[index] = x.T[index] - min_col
	return x


def load_dataset(path='./processed_dataset/data.csv', split=0.8, shuffle=True, seed=0):
	np.random.seed(seed)
	df = pd.read_csv(path)
	df = df.values
	if shuffle:
		np.random.shuffle(df)
	train = df[:int(df.shape[0]*split)]
	validation = df[int(df.shape[0]*split):]

	train_x, train_y = train.T[:12].T, train.T[12:].T
	validation_x, validation_y = validation.T[:12].T, validation.T[12:].T

	train_x, validation_x = min_max_x(train_x), min_max_x(validation_x)

	train_x, train_y, validation_x, validation_y = train_x.astype(np.float32), train_y.astype(np.float32), validation_x.astype(np.float32), validation_y.astype(np.float32)

	train_x, train_y, validation_x, validation_y = torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(validation_x), torch.from_numpy(validation_y)
	return train_x, train_y, validation_x, validation_y

if __name__ == '__main__':
	train_x, train_y, validation_x, validation_y = load_dataset()
	print(train_x.shape, train_y.shape, validation_x.shape, validation_y.shape)