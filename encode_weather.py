import torch
import numpy as np
import pandas as pd
from RBM import RBM
from DBN import DBN
from load_dataset import load_dataset

if __name__ == '__main__':
	train_x, train_y, validation_x, validation_y = load_dataset()

	layers = [10, 7, 5]
	dbn = DBN(12, layers, k=100)
	dbn.train_DBN(train_x)

	print('\n\n---Saving DBN---\n\n\n')
	layer_parameters = dbn.layer_parameters
	torch.save(layer_parameters, './saved_models/dbn.pt')

	print('Saving Model---\n\n\n')
	dbn.layer_parameters = torch.load('./saved_models/dbn.pt')
	model = dbn.initialize_model()
	torch.save(model, './saved_models/dbn_pretrained_model.pt')

	train_reconstructed_x, train_hidden_x = dbn.reconstructor(train_x)
	train_mae = torch.mean(torch.abs(train_reconstructed_x - train_x)).item()
	print("MAE Training Set:", train_mae)

	validation_reconstructed_x, validation_hidden_x = dbn.reconstructor(validation_x)
	validation_mae = torch.mean(torch.abs(validation_reconstructed_x - validation_x)).item()
	print("MAE Validation Set:", validation_mae)

	results = pd.DataFrame({'MAE TRAIN': [train_mae],	'MAE VALIDATION':[validation_mae]})
	results.to_csv('results.csv', index=False)