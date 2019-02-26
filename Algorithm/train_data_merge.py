# -*- coding: UTF-8 -*-
import pandas as pd
import os


def file_name(file_dir):
	temp_list = []
	for root, dirs, files in os.walk(file_dir):
		temp_list.append(dirs)
		temp_list.append(files)
	result_list = []
	if len(temp_list) > 4:
		result_list.append(temp_list[3])
		result_list.append(temp_list[5])
		return temp_list[0], result_list
	else:
		result_list.append(temp_list[3])
		return temp_list[0], result_list


def file_name_list(file_dir):
	for root, dirs, files in os.walk(file_dir):
		return files


def data_merge(file_dir, filename_list):
	filename = filename_list.pop()
	df = pd.read_csv("%s/%s" % (file_dir, filename))
	while filename_list != []:
		filename = filename_list.pop()
		df2 = pd.read_csv("%s/%s" % (file_dir, filename))
		df = df.append(df2, ignore_index=True)
	return df


if __name__ == '__main__':
	web_name = 'sina'
	file_dir = './TrainingData/%s' % web_name
	filename_list = file_name_list(file_dir)
	df = data_merge(file_dir,filename_list)
	print(df['label'].value_counts())
