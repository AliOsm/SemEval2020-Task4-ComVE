import csv
import time
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from os.path import join

from tqdm import tqdm
from keras.models import load_model, Input, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

def build_model(embeddings_size):
	train_embeddings_input = Input(shape=(embeddings_size,))

	dense = Dropout(args.dropout_rate)(
		Dense(
			units=512,
			activation='relu',
			kernel_initializer='he_normal'
		)(train_embeddings_input)
	)

	dense = Dropout(args.dropout_rate)(
		Dense(
			units=256,
			activation='relu',
			kernel_initializer='he_normal'
		)(dense)
	)

	dense = Dropout(args.dropout_rate)(
		Dense(
			units=128,
			activation='relu',
			kernel_initializer='he_normal'
		)(dense)
	)

	dense = Dropout(args.dropout_rate)(
		Dense(
			units=64,
			activation='relu',
			kernel_initializer='he_normal'
		)(dense)
	)

	output = Dense(
		units=1,
		activation='sigmoid',
		kernel_initializer='he_normal'
	)(dense)

	model = Model(train_embeddings_input, output)

	model.compile(
		optimizer=Adam(lr=0.001),
		loss='binary_crossentropy',
		metrics=['accuracy']
	)
	model.summary()

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
	parser.add_argument('--models-dir', default='models_dir')
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--batch-size', default=128, type=int)
	parser.add_argument('--dropout-rate', default=0.2, type=float)
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

	embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3')

	if not args.no_train:
		start_time = time.time()

		train_embeddings = list()
		with tf.Session() as session:
			session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

			for i in tqdm(range(0, len(train_sentences), 512)):
				train_embeddings.extend(session.run(embed(train_sentences[i:i + 512])))

		model = build_model(len(train_embeddings[0]))

		checkpoint_cb = ModelCheckpoint(
			filepath=join(args.models_dir, '4-universal_sentence_encoder-feed_forward_neural_network-epoch{epoch:02d}.h5')
		)

		model.fit(
			x=np.array(train_embeddings),
			y=np.array(train_labels),
			epochs=args.epochs,
			batch_size=args.batch_size,
			callbacks=[checkpoint_cb]
		)

		end_time = time.time()
		print('Training time:', end_time - start_time, 'seconds')

		train_predictions = model.predict(np.array(train_embeddings), batch_size=256).squeeze()
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
		model = load_model(join(args.models_dir, '4-universal_sentence_encoder-feed_forward_neural_network.h5'))

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

	dev_embeddings = list()
	with tf.Session() as session:
		session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

		for i in tqdm(range(0, len(dev_sentences), 512)):
			dev_embeddings.extend(session.run(embed(dev_sentences[i:i + 512])))

	dev_predictions = model.predict(np.array(dev_embeddings), batch_size=256).squeeze()

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

	with open(join(args.data_dir, '4-universal_sentence_encoder-feed_forward_neural_network.csv'), 'w') as file:
		writer = csv.writer(file)

		for i in range(0, len(dev_predictions), 2):
			if dev_predictions[i] > dev_predictions[i + 1]:
				writer.writerow([i // 2 + 1, 0])
			else:
				writer.writerow([i // 2 + 1, 1])
