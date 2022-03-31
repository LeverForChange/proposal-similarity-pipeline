import pandas as pd
import pickle

def neighbors_to_csv():
  path = 'data/'
  df = pd.read_csv(path + 'lfc-proposals-clean.csv')
  knn_indices = pickle.load(open(path + 'knn_indices.pkl', 'rb'))
  knn_distances = pickle.load(open(path + 'knn_distances.pkl', 'rb'))
  res = {'Proposal ID': [], 'Similar Proposal IDs': [], 'Distances': []}

  knn = list(zip(knn_indices, knn_distances))
  for indices, dists in knn:
    res['Proposal ID'].append(df.iloc[indices[0]]['ID'])
    if len(indices) > 1:
      res['Similar Proposal IDs'].append(','.join([df.iloc[x]['ID'] for x in indices[1:]]))
      res['Distances'].append(','.join([str(x) for x in dists[1:]]))
    else:
      res['Similar Proposal IDs'].append('')
      res['Distances'].append('')

  res = pd.DataFrame(res)
  res.to_csv('data/lfc-proposal-neighbors.csv', index=False)

if __name__ == '__main__':
  neighbors_to_csv()