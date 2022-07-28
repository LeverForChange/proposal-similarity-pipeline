import asyncio
import json
import time

import cleaner
import preprocessor
import apply_umap
import topic_model
import s3_uploader

def run(**kwargs):
  t0 = time.time()
    
  # Clean data
  if kwargs['cleaner']['run']:
    print('Cleaning data...')
    cleaner.run(**kwargs['cleaner'] | kwargs['global'])
    print('\n')

  # Preprocess
  if kwargs['preprocessor']['run']:
    print('Preprocessing data...')
    preprocessor.run(**kwargs['preprocessor'] | kwargs['global'])
    print('\n')

  # UMAP
  if kwargs['apply_umap']['run']:
    print('Running UMAP...')
    apply_umap.run(**kwargs['apply_umap'] | kwargs['global'])
    print('\n')

  # Topic model
  if kwargs['topic_model']['run']:
    print('Running topic modelling...')
    topic_model.run(**kwargs['topic_model'] | kwargs['global'])
    print('\n')

  # S3 upload
  if kwargs['s3_uploader']['run']:
    print('Uploading to S3...')
    s3_uploader.run(**kwargs['s3_uploader'] | kwargs['global'])
    print('\n')

  print('Data output files written to data/')
  print('Pipeline completed in', f'{round(time.time() - t0, 2)}s')

if __name__ == '__main__':
  kwargs = json.load(open('args.json'))
  run(**kwargs)