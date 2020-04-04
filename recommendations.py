import pandas as pd
import numpy as np
import argparse
from math import isnan
from pprint import pprint
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

parser = argparse.ArgumentParser()
parser.add_argument('-a', dest='entry', action='store', type=str, help='Anime title')
args = parser.parse_args()

anime_input = args.entry

df = pd.read_csv("C:\\Users\\gonza\\Documents\\myanimelist-dataset\\anime_cleaned.csv")
# df.info(memory_usage='deep')

df = df[['title','title_english','source', 'genre', 'score']]

# df["titles"] = df["title"] + " " + df['title_english']

df.set_index(['title_english'], inplace=True, drop=True)

df["categories"] = df["source"] + " " + df["genre"] + " " + df["score"].astype(str)

df.drop(columns = ['title', 'source', 'genre', 'score'], inplace = True)

count = CountVectorizer()
count_matrix = count.fit_transform(df['categories'].values.astype('U'))

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)

# fuzzy search on user input
# pprint(anime_input)
choices = (df.index.values).astype('str')

matches = process.extract(anime_input, choices, scorer=fuzz.ratio)
match_list = []

for match in matches:
    if match[1] > 65:
        match_list.append(match[0])
    
# pprint(match_list)

def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_animes = []

    idx = indices[indices == title].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:11].index)

    for i in top_10_indexes:
        recommended_animes.append(list(df.index)[i])
        
    return recommended_animes

# pprint(df.head())

results = []

for show in match_list:
    # pprint(recommendations(show))
    results = results + recommendations(show)

# pprint("\\n")
# pprint("\\n")
non_nan_results = results
for result in non_nan_results:
    if isinstance(result, str) == False:
        non_nan_results.remove(result)
res = []
[res.append(x) for x in non_nan_results if x not in res]
pprint(res)