import hdbscan
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def run(**kwargs):
  t0 = time.time()
  path = 'data/'
  embeddings = pickle.load(open(path + 'embeddings.pkl', 'rb'))
  df = pd.read_csv(path + 'lfc-proposals-clean.csv')

  # Apply HDBSCAN on the dimension reduced (UMAP) dataset to generate clusters
  clusters = hdbscan.HDBSCAN(
    min_cluster_size=kwargs['min_cluster_size'],
    cluster_selection_epsilon=kwargs['cluster_selection_epsilon'],
    alpha=kwargs['alpha']
  ).fit(embeddings)
  print(f'Generated {len(set(clusters.labels_))} clusters in', f'{round(time.time() - t0, 2)}s')

  # outlier scores (higher = more unique)
  df['Outlier Score'] = clusters.outlier_scores_

  # "central" cluster points
  exemplars = clusters.exemplars_

  # c-TF-IDF for each cluster
  # See: https://www.kdnuggets.com/2020/11/topic-modeling-bert.html
  df['Topic'] = clusters.labels_
  docs_by_topic = df.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

  def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count

  # Creates an "importance" value for each word in each cluster, used to create the topic label
  tf_idf, count = c_tf_idf(docs_by_topic['Document'].values, m=len(df))

  # Prune the topics to extract the most important vocab
  def extract_top_n_words_per_topic(tf_idf, count, docs_by_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_by_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

  topic_words = extract_top_n_words_per_topic(tf_idf, count, docs_by_topic)
  print('Cluster | Top Words')
  for cluster, words in topic_words.items():
    print(cluster, ', '.join([w[0] for w in words[:5]]))

  # Compile topic words and exemplars into a matrix
  topics = {}
  for cluster, words in topic_words.items():
    topic_data = {'words': [w[0] for w in words[:5]]}
    if cluster != -1:
      topic_data['exemplar'] = tuple(exemplars[cluster][0])
    topics[cluster] = topic_data
  pickle.dump(topics, open(path + 'topics.pkl', 'wb'))

  # re-save the CSV with outlier info
  df.to_csv(path + 'lfc-proposals-clean.csv')

if __name__ == '__main__':
  import json
  kwargs = json.load(open('args.json'))['topic_model']
  run(**kwargs)