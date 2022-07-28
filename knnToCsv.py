import pandas as pd
import pickle
import sys

def neighbors_to_csv(filename, modelTag):
  path = f'data/{modelTag}_'
  df = pd.read_csv('data/' + filename)
  knn = pickle.load(open(path + 'knn_indices.pkl', 'rb'))
  res = {'Proposal ID': [], 'Similar Proposal IDs': []}

  for i, indices in enumerate(knn):
    res['Proposal ID'].append(df.iloc[i]['ID'])
    if len(indices) > 1:
      res['Similar Proposal IDs'].append(','.join([df.iloc[x]['ID'] for x in indices[1:]]))
    else:
      res['Similar Proposal IDs'].append('')

  res = pd.DataFrame(res)
  res.to_csv(f'data/{modelTag}_neighbors.csv', index=False)

if __name__ == '__main__':
  filename = sys.argv[1]
  modelTag = sys.argv[2]
  neighbors_to_csv(filename, modelTag)