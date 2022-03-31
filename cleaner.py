import pandas as pd
import json
import html
import re
import time
from textblob import Word
from nltk.corpus import stopwords

def run(**kwargs):
  t0 = time.time()

  # Load data
  path = 'data/'
  df = pd.read_csv(path + 'lfc-proposals.csv')
  document_col = kwargs['document_col'] # Used for UMAP dimension reduction
  if not isinstance(document_col, list):
    document_col = [document_col]

  # Create unified unique ID col
  df['Competition Domain'] = df['Competition Domain'].fillna('MISSINGCOMP')
  df['ID'] = df.apply(
    lambda x: f"{x['Competition Domain']}-{x['Application #']}",
    axis=1
    )
  df.drop_duplicates(subset=['ID'], inplace=True)

  # Drop cols with empty project titles
  df.dropna(subset=[*document_col, 'Project Title'], inplace=True)

  # Clean Work Location
  def clean_location(x):
    try:
      x = json.loads(x)
      val = x['Country'] + ' | ' + x['State/Province']
      if x.get('City'):
        val += ' | ' + x['City']
      return val
    except:
      return ''

  if 'Future Work #1 Location' in df.columns:
    df['Future Work #1 Location'] = df['Future Work #1 Location'].apply(lambda x: clean_location(x))
  if 'Organization Location' in df.columns:
    df['Organization Location'] = df['Organization Location'].apply(lambda x: clean_location(x))

  # Clean text columns
  stop = stopwords.words('english')

  def clean_text(text):

    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = text.replace('<br/>', ' ')
    text = text.replace('<br>', ' ')
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('(<[^<>]{0,}>)', ' ', text)
    text = ' '.join(x for x in text.split() if x not in stop and len(x) > 1)
    text = ' '.join([Word(word).lemmatize() for word in text.split()])

    if not text:
      text = 'TEXT_DECODE_FAILURE'

    return text

  if isinstance(document_col, list):
    df['Document'] = df[document_col].agg('\n\n'.join, axis=1)
  df['Document'] = df['Document'].apply(lambda x: clean_text(x))

  # Sanitize text columns (for front-end)
  def sanitize_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub('(<[^<>]{0,}>)', ' ', text)
    return text

  df['Project Title'] = df['Project Title'].apply(lambda x: sanitize_text(x))
  df['Document Sanitized'] = df[document_col].apply(lambda row: ' '.join([sanitize_text(r) for r in row]), axis=1)
  if 'Organization Name' in df.columns:
   df['Organization Name'] = df['Organization Name'].apply(lambda x: sanitize_text(x))
  if 'Number of Employees' in df.columns:
    df['Number of Employees'] = df['Number of Employees'].apply(lambda x: sanitize_text(x))
  if 'Annual Operating Budget' in df.columns:
    df['Annual Operating Budget'] = df['Annual Operating Budget'].apply(lambda x: sanitize_text(x))

  # Clean Priority Pops
  def clean_priority_populations(val):
    try:
      val = val.split(',')
      val = ', '.join(val)
    except:
      pass
    return val

  if 'Priority Populations' in df.columns:
    df['Priority Populations'] = df['Priority Populations'].apply(lambda x: clean_priority_populations(x))

  # Clean Projected Costs
  def clean_projected_costs(val):
    if pd.isnull(val) or 'nan' in str(val):
      return ''
    elif isinstance(val, str):
      if ','  in val:
        return '$' + val
      else:
        try:
          val = int(val)
          val = '{:,}'.format(val)
        except:
          pass
        finally:
          return '$' + val
    elif isinstance(val, float) or isinstance(val, int):
      val = '{:,}'.format(val)
      return '$' + val

  if 'Total Projected Costs' in df.columns:
    df['Total Projected Costs'] = df['Total Projected Costs'].apply(lambda x: clean_projected_costs(x))

  # Clean Primary Subject Area
  def clean_primary_subject_area(val):
    try:
      val = json.loads(val)
      return val.get('Preferred Label')
    except:
      return val
    
  if 'Primary Subject Area' in df.columns:
    df['Primary Subject Area'] = df['Primary Subject Area'].apply(lambda x: clean_primary_subject_area(x))

  df.drop(columns=document_col, inplace=True)

  df.to_csv(path + 'lfc-proposals-clean.csv', index=False)
  print('Cleaned data in', f'{round(time.time() - t0, 2)}s')
  print('Remaining proposals after cleaning:', len(df))

if __name__ == '__main__':
  import json
  kwargs = json.load(open('args.json'))['cleaner']
  run(**kwargs)