# LFC Proposal Data Model Pipeline

Customizable script to generate 3D embeddings, topic labels, and a neighbor matrix using Lever for Change proposal data.

For use with non-LfC data, implement your own cleaning function.

## General Instructions

1. `pip install -r requirements.txt` In case problems arise, consult: https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst
2. The NLP tasks require some external datasets to be downloaded on the local machine. 
  - Run the following command in your terminal: `python -m textblob.download_corpora`. If you encounter issues, consult: https://stackoverflow.com/questions/41310885/error-downloading-textblob-certificate-verify-failed
  - Then, open a python instance and run: `import nltk`, then `nltk.download('stopwords')`, then `nltk.download('omw-1.4')`. If you have problems, consult here: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
  - If more problems persist, you may need to download nltk corpora "al la carte". Follow the python shell commands outputted in red by nltk.
3. `args.json` contains keyword arguments for each step in the pipeline. Stages can be added or dropped by setting the `run` boolean for each stage. For example, if a cleaned and preprocessed dataset already exists, you may want to set all  `run` flags to `false` except for `apply_umap` and `topic_model`. I would recommend copying this file and naming it `args.local.json`, which will be ignored by git. This way, the original config can be preserved.
4. `python pipeline.py` - output files will be placed under `data/`.
5. To upload the output files to S3 - a requirement for the hosted webapp to run with updated data, don't forget to cycle the container - ensure the `s3_uploader` run flag is set to `true`. You'll need to run `aws configure` in your terminal first to set up the AWS environment locally. Ensure the `AWS_ACCESS_KEY` and `AWS_ACCESS_SECRET` are up to date in the container's environment variables.

A small function to convert the KNN indices matrix to a CSV can be found in `knnToCsv.py`. This is useful for recalculating similar proposals for ingestion into Global View.

## UMAP + Topic Model Pipeline Arguments

Required fields for downloaded Torque file, if using the pipeline to regenerate embeddings for the landscape app. For generic use, implement your own cleaning step, and adjust the `document_col` parameter.

```
"Primary Subject Area",
"Total Projected Costs",
"Priority Populations",
"Future Work Locations",
"Annual Operating Budget",
"Number of Employees",
"Organization Name", 
"Organization Location",

"Executive Summary"
```

It is also recommended to exclude `100Change2017` from the downloaded proposals.

### Global Parameters

1. `input_file_name` Name of the input CSV file to run the pipeline on.
2. `output_file_name` The name of the cleaned CSV file outputted in the cleaning step.
3. `model_tag` A prefx that will be appended to each output file, useful for identifying across multiple pipeline runs.

### Cleaner

The `cleaner.py` module cleans the data downloaded from Torque.

1. `document_col` This field specifies which field to turn into the `Document` column in the dataframe. This field will become the one used to perform the TF-IDF vectorization and subsequent dimension reduction on.

### Preprocessing

The `preprocessor.py` module transforms the `Document` column of the data into a TF-IDF vector matrix.
See more descriptions of argument values [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

Currently `Document` is hardcoded to be the only feature evaluated in the UMAP model, but this can easily be extended to include other text, categorical, or numeric fields (see the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) docs). 

I've found that the Executive Summary provides the best output with the remaining fields useful for filtering/grouping the output.

### UMAP

The `apply-umap.py` module fits the preprocessed data to a UMAP model and produces a 2D matrix of KNN indices.

1. `metric` this is just for the UMAP step, the neighbor finding step will always use `euclidean`
2. `components` the number of components to reduce to
3. `min_dist` the minimum distance between two points, below which UMAP will disregard in dimension reduction
4. `densmap` turning this on leads to clumpier data
5. `n_neighbors` the number of neighbors to consider during UMAP computation
6. `threshold` the euclidean distance below which an external node is considered a neighbor or not, higher vaues result in more neighbors per point
7. `neighbor_approach` whether to find a mapping of neighbors for each proposal based on N nearest neighbors or any neighbor within a radius. Options are either `radius` or `knn`. If `radius` the `threshold` value will be used, if `knn` the `n_neighbors` value will be used. `radius` is recommended for building an embedding for the proposal landscape app, while `knn` is better suited for identifying similar proposals in Global View.

Learn more in the [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html) and the [KNN documentation](https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.nearest_neighbors)

### Topic Model

The `topic-model.py` module takes the dimension-reduced data from UMAP and finds clusters in the data space. It then generates a list of labeled clusters (by most important words) and their associated cluster centers. The module uses [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html), which allows for data points to be assigned `-1` as a cluster - in other words not belonging to any cluster. For more information on the topic modelling approach, see [here](https://www.kdnuggets.com/2020/11/topic-modeling-bert.html). See the [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) for descriptions of the arguments.

### S3 Uploader

The `s3_uploader.py` module puts every file found in `data/` into an S3 bucket. You'll need to create your own AWS account, S3 bucket, and AWS access key. 

1. `bucket` the name of the bucket housing your model data
2. `acl` the default ACL for each file uploaded, recommend `private` so that objects can only be accesse with access keys

## Data Specifications

### Proposal Dataframe

Pandas DataFrame of cleaned proposal data stored in `consts.py` as `df` in the web app repository and is outputted as a CSV with the name defined in `output_file_name`. 

Note that the `Application #` column is not the index corresponding to the proposal in the loaded dataframe, the index is a different value internally generated by pandas.

The topic model module adds two columns to the cleaned dataset: `Topic` and `Outlier Score`. `Topic` is a numeric variable corresponsing to the index of the topic specified in `topics.pkl`. `Outlier Score` is a numeric variable denoting how closely the observation is associated to its assigned topic. Higher values mean that the observation is more loosely associated with its cluster, and is therefore more of a unique/outlier observation.

### UMAP Embeddings

An embedding matrix with coordinates for each proposal on each dimension in _[1, components]_. Stored in `consts.py` as `embeddings` in the web app repository and is outputted in `data/` as `<MODEL_TAG>_embeddings.pkl`.

With the default number of components (3), the matrix simply becomes a list of x, y, z coordinates. The web app will need minor tweaks to support 2D coordinate pairs or certain subsets of higher-dimensional data.

```
[
  (3.2345, 1.009, 0.13677),
  (6.32454, 2.083, 1.562),
  ...,
  (4.09092, 3.0004, 0.158)
]
```

### KNN Indices Matrix

Stored in `consts.py` as `knn_indices` in the web app repository and is outputted in `data/` as `<MODEL_TAG>_knn_indices.pkl`.

For each element in the KNN Indices Matrix, the index of the element corresponds to a proposal at index `i` in the Proposal Dataframe. Each element contains a list of indices, which represent neighbors of that proposal. A proposal may have no neighbors (empty list).

For example:

```
[
  [0, 10, 15],
  [],
  ...,
  [5000]
]
```

These indices might evaluate to something like (made up names):

```
[
  [Health Equity in Chicago, Building a Hospital in Indiana, Advocacy for Children of Cancer Patients],
  [],
  ...,
  [Investing in Guatemalan Science Education]
]
```

### Topic Model Data

Stored in `consts.py` as `topics` in the web app repository and in `data/` as `<MODEL_TAG>_topics.pkl`.

This is a dictionary of `cluster_label -> topic_information` mappings. The first key is always `-1`, which corresponds to the "non-cluster" entry and therefore does not have an `exemplar` entry. All other entries have a set of `words` and a 3D `exemplar` coordinate, denoting the coordinates of the point which represents the best center of the cluster.

```
{
  -1: {
    'words': ['social', 'economic', 'work', 'education', 'poverty']
  },
  0: {
    'words': ['news', 'newsroom', 'medium', 'journalism', 'journalist'], 
    'exemplar': (7.306426048278809, -0.29259100556373596, -2.392807960510254)
  }
}
```

The keys in the topics dictionary map to the `Topic` column in the Proposal Dataframe.

## KNN to CSV utility function

For convenience, a utility function is provided to map proposals to a list of neighboring proposals. The CSV generated has two columns, the first containing the ID of a target proposal, and the second a comma separated list of neighboring proposals (or blank if no neighbors exist). In other words, this is a reformatted CSV version of the KNN Indices Matrix.

#### Arguments

Example command: `python knnToCsv.py LFC_proposals-clean.csv LFC`

Args:
1. `filename` The name of the cleaned CSV generated in the cleaning step of the pipeline
2. `modelTag` The model tag of the desired pipeline output - determines which knn_indicies matrix to select