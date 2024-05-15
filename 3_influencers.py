import networkx as nx
import os
import pandas as pd
import ast
from tqdm import tqdm

authors = dict()

G = nx.read_gexf('data/test.2024wallstreetbets.gexf')

df = pd.read_csv('final.2024wallstreetbets.csv')

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row['Author'] not in authors:
        authors[row['Author']] = 0
    authors[row['Author']] += 1

print(len(df), len(authors))

# топ-100 авторов по кол-ву публикаций
InflPublications = {k: v for k, v in sorted(authors.items(), key=lambda item: item[1], reverse=True)[:100]}
print(InflPublications)

# рассчитывание значений PageRank
pagerankdict = nx.pagerank(G)
print()
# топ-100 авторов с наибольшим значением PageRank
InflPageRank = {k: v for k, v in sorted(pagerankdict.items(), key=lambda item: item[1], reverse=True)[:100]}
print(InflPageRank)


# рассчитывание значений in-degree
indegree_map = {v: d for v, d in G.in_degree() if d > 0}
print()
# топ-100 авторов с наибольшим значением in-degree
InflinDegree = {k: v for k, v in sorted(indegree_map.items(), key=lambda item: item[1], reverse=True)[:100]}
print(InflinDegree)

# рассчитывание значений out-degree
outdegree_map = {v: d for v, d in G.out_degree() if d > 0}
print()
# топ-100 авторов с наибольшим значением out-degree
InfloutDegree = {k: v for k, v in sorted(outdegree_map.items(), key=lambda item: item[1], reverse=True)[:100]}
print(InfloutDegree)


infldf = pd.DataFrame()

# рассчитывание значений betweenness
btwnnss = nx.betweenness_centrality(G)
print()
# топ-100 авторов с наибольшим значением betweenness
InflBtwnnss = {k: v for k, v in sorted(btwnnss.items(), key=lambda item: item[1], reverse=True)[:100]}
print(InflBtwnnss)

print()

InflTOP = list(InflPageRank.keys() & InflBtwnnss.keys()) # составление списка инфлюенсеров
alltop = list(InflPublications.keys() & InflPageRank.keys() & InfloutDegree.keys() & InflinDegree.keys() & InflBtwnnss.keys()) # пользователи вошедшие в топ-100 по всем мерам центральности
print(InflTOP)
print(alltop)

with open("InflTOP.WSB.txt", "w") as output:
    for listitem in InflTOP:
        output.write(f'{listitem}\n')

# составление датасета пользователь-PageRank-betweenness
weights = pd.DataFrame()
for author in authors:
    try:
        weights = pd.concat([weights, pd.DataFrame([{
                    'author': author,
                    'PageRank': pagerankdict[author],
                    'Betweenness': btwnnss[author]
                }])], ignore_index=True)
    except:
        None

weights.to_csv('weights.WSB.csv')

partition = ast.literal_eval(open('clusters.txt', 'r').read())

# составление датасета инфлюенсеров
for author in InflTOP:
    publications = df.loc[df['Author'] == author]
    numbpub = publications.shape[0]
    numbposts = publications.loc[publications['type'] == 'post'].shape[0]
    numbcom = publications.loc[publications['type'] == 'comment'].shape[0]
    inalltop = author in alltop
    ups = publications['ups'].sum(axis=0)
    score = publications['score'].sum(axis=0) / publications.shape[0]
    if numbposts != 0:
        upvote_ratio = publications['upvote_ratio'].sum(axis=0) / numbposts
    else:
        upvote_ratio = 0

    selftickers = dict()
    for index, row in publications.iterrows():
        orgs = ast.literal_eval(row['ORG'])
        for org in orgs:
            if org not in selftickers:
                selftickers[org] = {row['id']}
            else:
                selftickers[org].add(row['id'])

    parenttickers = dict()
    for index, row in publications.iterrows():
        parentid = row['Parent id']
        if parentid != '0':
            ptickers = ast.literal_eval(df.loc[df['id'] == parentid, 'ORG'].iloc[0])
            for ticker in ptickers:
                if ticker not in parenttickers:
                    parenttickers[ticker] = {parentid}
                else:
                    parenttickers[ticker].add(parentid)
    
    for i in range(len(partition)):
        if author in partition[i]:
            cluster = {i}
            break

    infldf = pd.concat([infldf, pd.DataFrame([{
                'author': author,
                'in AllTOP': inalltop,
                'publications': numbpub,
                'posts': numbposts,
                'comments': numbcom,
                'in-Degree': indegree_map[author],
                'PageRank': pagerankdict[author],
                'Betweenness': btwnnss[author],
                'upvote_ratio': upvote_ratio,
                'ups': ups,
                'score': score,
                'self tickers': selftickers,
                'parent tickers': parenttickers,
                'cluster ind': cluster
            }])], ignore_index=True)

infldf.to_csv('influencers.WSB.csv')
