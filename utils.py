import pandas as pd
import pickle

def neighbors_to_csv():
  path = 'data/'
  df = pd.read_csv(path + 'lfc-proposals-clean.csv')
  knn = pickle.load(open(path + 'knn_indices.pkl', 'rb'))
  res = {'Proposal ID': [], 'Similar Proposal IDs': []}

  for i, indices in enumerate(knn):
    res['Proposal ID'].append(df.iloc[i]['ID'])
    if len(indices) > 1:
      res['Similar Proposal IDs'].append(','.join([df.iloc[x]['ID'] for x in indices[1:]]))
    else:
      res['Similar Proposal IDs'].append('')

  res = pd.DataFrame(res)
  res.to_csv('data/lfc-proposal-neighbors.csv', index=False)

if __name__ == '__main__':
  neighbors_to_csv()