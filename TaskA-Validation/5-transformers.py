import csv
import time
import argparse

import pandas as pd

from os.path import join

from simpletransformers.model import TransformerModel

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
	parser.add_argument('--models-dir', default='models_dir')
	parser.add_argument('--model-name', required=True)
	parser.add_argument('--max-seq-length', default=30, type=int)
	parser.add_argument('--train-batch-size', default=32, type=int)
	parser.add_argument('--eval-batch-size', default=32, type=int)
	parser.add_argument('--num-train-epochs', default=10, type=int)
	parser.add_argument('--overwrite-output-dir', action='store_false')
	parser.add_argument('--reprocess-input-data', action='store_false')
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

	train_data = list()
	for train_sentence, train_label in zip(train_sentences, train_labels):
		train_data.append([train_sentence, train_label])
	train_data = pd.DataFrame(train_data)

	model_type = args.model_name.split('-')[0]
	model_name = args.model_name

	if args.no_train:
		model_name = join(args.models_dir, args.model_name)

	model_args = {
		'output_dir': join(args.models_dir, args.model_name),

		'fp16': False,
		'max_seq_length': args.max_seq_length,
		'train_batch_size': args.train_batch_size,
		'eval_batch_size': args.eval_batch_size,
		'num_train_epochs': args.num_train_epochs,

		'save_steps': 1000000000,

		'overwrite_output_dir': args.overwrite_output_dir,
		'reprocess_input_data': args.reprocess_input_data
	}

	model = TransformerModel(
		model_type,
		model_name,
		args=model_args
	)

	if not args.no_train:
		start_time = time.time()

		model.train_model(train_data)

		end_time = time.time()
		print('Training time:', end_time - start_time, 'seconds')

		_, train_predictions = model.predict(train_sentences)

		correct = 0
		for i in range(0, len(train_labels), 2):
			if train_predictions[i][1] > train_predictions[i + 1][1]:
				if train_labels[i] == 1 and train_labels[i + 1] == 0:
					correct += 1
			else:
				if train_labels[i] == 0 and train_labels[i + 1] == 1:
					correct += 1
		print('Training accuracy: {0:0.4f}%'.format(correct / (len(train_labels) / 2) * 100))

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

	dev_data = list()
	for dev_sentence, dev_label in zip(dev_sentences, dev_labels):
		dev_data.append([dev_sentence, dev_label])
	dev_data = pd.DataFrame(dev_data)

	start_time = time.time()

	_, dev_predictions = model.predict(dev_sentences)

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

	with open(join(args.data_dir, '{}.csv'.format(args.model_name)), 'w') as file:
		writer = csv.writer(file)

		for i in range(0, len(dev_predictions), 2):
			if dev_predictions[i][1] > dev_predictions[i + 1][1]:
				writer.writerow([i // 2 + 1, 0])
			else:
				writer.writerow([i // 2 + 1, 1])
