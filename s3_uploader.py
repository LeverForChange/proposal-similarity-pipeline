import boto3
import os
import time
import json

def run(**kwargs):
  s3 = boto3.client('s3')

  path = f'data/{kwargs["model_tag"]}_'
  for f in os.listdir(path):
    key = os.path.join(path, f)
    if os.path.isfile(key):
      t0 = time.time()
      s3.put_object(
        Body=open(key, 'rb'),
        Bucket=kwargs['bucket'],
        Key=key.split('/')[-1],
        ACL=kwargs['acl']
        )
      print('Uploaded:', key.split('/')[-1], 'in', f'{round(time.time() - t0, 2)}s')

if __name__ == '__main__':
  kwargs = json.load(open('args.json'))
  run(**kwargs['s3_uploader'] | kwargs['global'])