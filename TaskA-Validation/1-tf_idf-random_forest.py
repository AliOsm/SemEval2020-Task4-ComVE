import csv
import time
import argparse

import pickle as pkl

from os.path import join

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

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

	if not args.no_train:
		start_time = time.time()

		classifier = Pipeline([
			('vect', CountVectorizer(max_features=10000, lowercase=False)),
			('tfidf', TfidfTransformer()),
			('clf', RandomForestClassifier(n_estimators=100, n_jobs=6))
		])

		classifier = classifier.fit(train_sentences, train_labels)

		end_time = time.time()
		print('Training time:', end_time - start_time, 'seconds')

		train_predictions = classifier.predict_proba(train_sentences)
		correct = 0
		for i in range(0, len(train_labels), 2):
			if train_predictions[i][1] > train_predictions[i + 1][1]:
				if train_labels[i] == 1 and train_labels[i + 1] == 0:
					correct += 1
			else:
				if train_labels[i] == 0 and train_labels[i + 1] == 1:
					correct += 1
		print('Training accuracy: {0:0.4f}%'.format(correct / (len(train_labels) / 2) * 100))
	else:
		with open(join(args.models_dir, '1-tf_idf-random_forest.pkl'), 'rb') as file:
			classifier = pkl.load(file)

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

	dev_predictions = classifier.predict_proba(dev_sentences)

	end_time = time.time()
	print('Evaluation time:', end_time - start_time, 'seconds')

	correct = 0
	for i in range(0, len(dev_labels), 2):
		if dev_predictions[i][1] > dev_predictions[i + 1][1]:
			if dev_labels[i] == 1 and dev_labels[i + 1] == 0:
				correct += 1
		else:
			if dev_labels[i] == 0 and dev_labels[i + 1] == 1:
				correct += 1
	print('Development accuracy: {0:0.4f}%'.format(correct / (len(dev_labels) / 2) * 100))

	with open(join(args.data_dir, '1-tf_idf-random_forest.csv'), 'w') as file:
		writer = csv.writer(file)

		for i in range(0, len(dev_predictions), 2):
			if dev_predictions[i][1] > dev_predictions[i + 1][1]:
				writer.writerow([i // 2 + 1, 0])
			else:
				writer.writerow([i // 2 + 1, 1])

	with open(join(args.models_dir, '1-tf_idf-random_forest.pkl'), 'wb') as file:
		pkl.dump(classifier, file)
