import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class PreProcess:
	def __init__(self, base='./dataset/'):
		self.years = sorted([base+i for i in os.listdir(base)])
		self.to_process_features = ['Longitude (x)', 'Latitude (y)', 
		'Year', 'Month', 'Day', 'Time', 'Temp (°C)',
		'Dew Point Temp (°C)', 'Rel Hum (%)',
		'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
		self.dataset = []
	
	def iterator(self, year):
		year = year+'/'
		files = [year+i for i in os.listdir(year)]
		for file in files:
			df = pd.read_csv(file, usecols=self.to_process_features)
			features = df.columns
			df = df.values
			for row in df:
				self.dataset.append(row)
		
	def progression(self):
		for year in tqdm(self.years):
			self.iterator(year)
		self.dataset = np.array(self.dataset)

		index_without_nan = []
		for index, row in enumerate(self.dataset):
			if np.any(row[:-1]!=row[:-1]):
				continue
			index_without_nan.append(index)

		self.dataset = self.dataset[index_without_nan]
		self.dataset = pd.DataFrame(self.dataset)		
		self.dataset.columns = self.to_process_features
		self.dataset = self.dataset.fillna("missing")
		return self.dataset

	def preprocess(self):
		self.dataset = self.progression()
		features = self.dataset.columns
		self.dataset = self.dataset.values
		x = self.dataset.T[:-1]
		y = self.dataset.T[-1]
		weather_divs = []
		uniques = []
		for row in y:
			row = [j.strip() for j in row.split(',')]
			weather_divs.append(row)
			uniques += row
		unique_weathers = 0
		uniques = list(set(uniques))

		y = []
		for row in weather_divs:
			row_new = [0]*len(uniques)
			for w in row:
				row_new[uniques.index(w)] = 1
			y.append(row_new)
		y = np.array(y)
		self.dataset = np.concatenate((x, y.T), axis=0).T
		time = np.array([int(i[:i.index(':')]) for i in self.dataset.T[5]])
		self.dataset.T[5] = time
		self.dataset = pd.DataFrame(self.dataset)
		self.dataset.columns = [i for i in features[:-1]] + [i for i in uniques]
		return self.dataset

if __name__ == '__main__':
	pp = PreProcess()
	dataset_processed = pp.preprocess()
	dataset_processed.to_csv('./processed_dataset/data.csv', index=False)