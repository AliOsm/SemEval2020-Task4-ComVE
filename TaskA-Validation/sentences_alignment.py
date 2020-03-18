import csv
import random
import argparse

from os.path import join

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def align(example):
	sentence1 = example[0]
	sentence2 = example[1]
	prediction = example[2]

	i1 = 0
	j1 = 0
	while i1 < len(sentence1) and j1 < len(sentence2) and sentence1[i1] == sentence2[j1]:
		i1 += 1
		j1 += 1

	i2 = len(sentence1) - 1
	j2 = len(sentence2) - 1
	while i2 >= 0 and j2 >= 0 and sentence1[i2] == sentence2[j2]:
		i2 -= 1
		j2 -= 1
	i2 += 1
	j2 += 1

	sentence1_p1 = ' '.join(sentence1[:i1])
	sentence1_p2 = ' '.join(sentence1[i1:i2])
	if len(sentence1_p1) != 0:
		sentence1_p2 = ' ' + sentence1_p2
	sentence1_p3 = ' '.join(sentence1[i2:])
	if len(sentence1_p2) != 0:
		sentence1_p3 = ' ' + sentence1_p3

	sentence2_p1 = ' '.join(sentence2[:j1])
	sentence2_p2 = ' '.join(sentence2[j1:j2])
	if len(sentence2_p1) != 0:
		sentence2_p2 = ' ' + sentence2_p2
	sentence2_p3 = ' '.join(sentence2[j2:])
	if len(sentence2_p2) != 0:
		sentence2_p3 = ' ' + sentence2_p3

	if prediction == 1:
		print(bcolors.OKGREEN + sentence1_p1 + bcolors.WARNING + sentence1_p2 + bcolors.OKGREEN + sentence1_p3 + bcolors.ENDC)
		print(sentence2_p1 + bcolors.WARNING + sentence2_p2 + bcolors.ENDC + sentence2_p3)
	else:
		print(sentence1_p1 + bcolors.WARNING + sentence1_p2 + bcolors.ENDC + sentence1_p3)
		print(bcolors.OKGREEN + sentence2_p1 + bcolors.WARNING + sentence2_p2 + bcolors.OKGREEN + sentence2_p3 + bcolors.ENDC)

	return max(i2 - i1, j2 - j1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
	args = parser.parse_args()

	data = list()

	with open(join(args.data_dir, 'training-x.csv'), 'r') as file:
		reader = csv.reader(file)
		next(reader)

		for row in reader:
			data.append([row[1].lower().split(), row[2].lower().split()])

	with open(join(args.data_dir, 'training-y.csv'), 'r') as file:
		reader = csv.reader(file)

		for idx, row in enumerate(reader):
			data[idx].append(int(row[1]))

	changes = dict()
	for idx, example in enumerate(data):
		print('Example number {}'.format(idx + 1))
		change = align(example)
		if change < 0: change = 0
		if change in changes:
			changes[change] += 1
		else:
			changes[change] = 1
		print('')

	print('Examples per number of differences')
	for key, value in sorted(changes.items()):
		if key < 0: continue
		print('{0:2d} difference(s): {1}'.format(key, value))
