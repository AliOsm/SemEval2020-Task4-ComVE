import csv
import argparse

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from os.path import join

from tqdm import tqdm
from sklearn.manifold import TSNE

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
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

	train_embeddings = list()
	with tf.Session() as session:
		session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

		for i in tqdm(range(0, len(train_sentences), 512)):
			train_embeddings.extend(session.run(embed(train_sentences[i:i + 512])))

	tsne_model = TSNE(perplexity=40, random_state=40, n_components=2, init='pca')
	train_embeddings = tsne_model.fit_transform(train_embeddings)

	x = []
	y = []
	for value in train_embeddings:
		x.append(value[0])
		y.append(value[1])

	fig, ax = plt.subplots(figsize=(16, 8))
	for i in range(len(x)):
		plt.scatter(
			x[i],
			y[i],
			c= '#111111' if train_labels[i] == 0 else '#aaaaaa'
		)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(16)

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)

	plt.tight_layout()
	plt.show()
