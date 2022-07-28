import pandas as pd
import pickle
import json

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def run(**kwargs):

  # Train/Test Split
  path = f'data/{kwargs["model_tag"]}_'
  train = pd.read_csv(path + kwargs['output_file_name'])
  
  # Declare TF-IDF vectorizer
  tfidf = TfidfVectorizer(
    ngram_range=(kwargs['ngram_range_min'], kwargs['ngram_range_max']),
    min_df=kwargs['min_df'],
    max_df=kwargs['max_df'],
    max_features=kwargs['max_features'],
    binary=kwargs['binary']
    )
  text_transformer = (f'tfidf-Document', tfidf, 'Document')

  # Assemble pre-processor
  preprocess = ColumnTransformer(
    remainder='drop',
    transformers=[text_transformer],
    verbose=True
    )
  preprocess_df = preprocess.fit_transform(train) #.toarray()

  pickle.dump(preprocess_df, open(path + 'preprocess_df.pkl', 'wb'))

if __name__ == '__main__':
  kwargs = json.load(open('args.json'))
  run(**kwargs['preprocessor'] | kwargs['global'])