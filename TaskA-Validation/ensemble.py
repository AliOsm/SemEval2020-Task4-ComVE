import csv
import argparse

from os.path import join

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir/testing')
	args = parser.parse_args()

	prediction_files = [
		'7-bert-large-cased.csv',
		'11-roberta-base.csv',
		'12-roberta-large.csv',
		'13-albert-xxlarge-v2.csv',
		'13-albert-xxlarge-v2.csv'
	]

	predictions = dict()
	for prediction_file in prediction_files:
		with open(join(args.data_dir, prediction_file), 'r') as file:
			reader = csv.reader(file)

			for row in reader:
				if row[0] in predictions:
					predictions[row[0]] += int(row[1])
				else:
					predictions[row[0]] = int(row[1])

	with open(join(args.data_dir, 'ensemble.csv'), 'w') as file:
		writer = csv.writer(file)

		for prediction in predictions:
			if predictions[prediction] >= len(prediction_files) - predictions[prediction]:
				writer.writerow([prediction, 1])
			else:
				writer.writerow([prediction, 0])
