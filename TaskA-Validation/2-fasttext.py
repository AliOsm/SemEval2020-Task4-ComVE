import os
import csv
import time
import argparse

import fasttext

from os.path import join

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
	parser.add_argument('--models-dir', default='models_dir')
	parser.add_argument('--no-train', action='store_true')
	args = parser.parse_args()

	train_sentences = list()
	train_labels = list()

	with open(join(args.data_dir, 'training-x.csv'), 'r') as file:
		reader = csv.reader(file)
		next(reader)

		for row in reader:
			train_sentences.append(row[1].strip())
			train_sentences.append(row[2].strip())

	with open(join(args.data_dir, 'training-y.csv'), 'r') as file:
		reader = csv.reader(file)

		for row in reader:
			train_labels.append(int(int(row[1]) == 0))
			train_labels.append(int(int(row[1]) == 1))

	with open(join(args.data_dir, 'fasttext_train'), 'w') as file:
		for train_sentence, train_label in zip(train_sentences, train_labels):
			file.write('__label__{} {}\n'.format(train_label, train_sentence))

	if not args.no_train:
		start_time = time.time()

		classifier = fasttext.train_supervised(join(args.data_dir, 'fasttext_train'), epoch=100, wordNgrams=3)

		end_time = time.time()
		print('Training time:', end_time - start_time, 'seconds')

		train_predictions = list()
		for train_sentence in train_sentences:
			train_prediction = list(zip(*classifier.predict(train_sentence, k=2)))
			train_prediction = sorted(train_prediction, key=lambda item: item[0])
			train_predictions.append(train_prediction[1][1])

		correct = 0
		for i in range(0, len(train_labels), 2):
			if train_predictions[i] > train_predictions[i + 1]:
				if train_labels[i] == 1 and train_labels[i + 1] == 0:
					correct += 1
			else:
				if train_labels[i] == 0 and train_labels[i + 1] == 1:
					correct += 1
		print('Training accuracy: {0:0.4f}%'.format(correct / (len(train_labels) / 2) * 100))
	else:
		classifier = fasttext.load_model(join(args.models_dir, '2-fasttext.ftz'))

	dev_sentences = list()
	dev_labels = list()

	with open(join(args.data_dir, 'development-x.csv'), 'r') as file:
		reader = csv.reader(file)
		next(reader)

		for row in reader:
			dev_sentences.append(row[1].strip())
			dev_sentences.append(row[2].strip())

	with open(join(args.data_dir, 'development-y.csv'), 'r') as file:
		reader = csv.reader(file)

		for row in reader:
			dev_labels.append(int(int(row[1]) == 0))
			dev_labels.append(int(int(row[1]) == 1))

	start_time = time.time()

	dev_predictions = list()
	for dev_sentence in dev_sentences:
		dev_prediction = list(zip(*classifier.predict(dev_sentence, k=2)))
		dev_prediction = sorted(dev_prediction, key=lambda item: item[0])
		dev_predictions.append(dev_prediction[1][1])

	end_time = time.time()
	print('Evaluation time:', end_time - start_time, 'seconds')

	correct = 0
	for i in range(0, len(dev_labels), 2):
		if dev_predictions[i] > dev_predictions[i + 1]:
			if dev_labels[i] == 1 and dev_labels[i + 1] == 0:
				correct += 1
		else:
			if dev_labels[i] == 0 and dev_labels[i + 1] == 1:
				correct += 1
	print('Development accuracy: {0:0.4f}%'.format(correct / (len(dev_labels) / 2) * 100))

	with open(join(args.data_dir, '2-fasttext.csv'), 'w') as file:
		writer = csv.writer(file)

		for i in range(0, len(dev_predictions), 2):
			if dev_predictions[i] > dev_predictions[i + 1]:
				writer.writerow([i // 2 + 1, 0])
			else:
				writer.writerow([i // 2 + 1, 1])

	classifier.save_model(join(args.models_dir, '2-fasttext.ftz'))

	os.remove(join(args.data_dir, 'fasttext_train'))
