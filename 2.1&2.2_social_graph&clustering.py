import csv
from tqdm import tqdm
import networkx as nx
from networkx import community
import pandas as pd
import math
import os

# создание социального графа пользователей -- ориентированный взвешенный граф, узлами которого являются пользователи

df = pd.read_csv('2024wallstreetbets.csv')

data = csv.reader(open('2024wallstreetbets.csv', encoding="utf8"))
headers = next(data)
data = list(data)
G = nx.DiGraph()
ind = 1

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    
    # пустое поле никнейма означает, что автор удалён
    # узлы пустых пользователей добавляются с целью сохранения структуры дискуссии
    if row['Author'] != '' and isinstance(row['Author'], str):
        name = row['Author']
    else:
        name = '@uthor' + str(ind)
        df.at[index, 'Author'] = name
        ind += 1

    # пост (публикация верхнего уровня) ни на кого не ссылается, поэтому такого пользователя просто добавляем в граф
    if row['type'] == 'post':
        if not G.has_node(name):
            G.add_node(name)
    
    # комментарий уже отображает связь между пользователями, поэтому добавляем пользователя, если он встречается в первый раз
    # далее если существует связь между родительской публикацией и текущим пользователем, то увеличиваем её вес на 1,
    # либо создаём новое ребро, если раньше его не было
    else:
        postauthor = df[df['id'] == row['post id']].iloc[0]['Author']
        parentauthor = df[df['id'] == row['Parent id']].iloc[0]['Author']
        if not G.has_node(name):
            G.add_node(name)
        """
        if G.has_edge(name, postauthor):
            G[name][postauthor]['weight'] += 1
        elif name != postauthor:
            try:
                math.isnan(postauthor)
            except TypeError:
                G.add_edge(name, postauthor, weight = 1)
        """
        if G.has_edge(name, parentauthor):
            G[name][parentauthor]['weight'] += 1
        elif name != parentauthor:
            try:
                math.isnan(parentauthor)
            except TypeError:
                G.add_edge(name, parentauthor, weight = 1)

# кластеризация графа, применяя алгоритм Лувейна
partition = community.louvain_communities(G)
print("Cnt communities: " + str(len(partition)))

f = open("clusters.txt", "w")
f.write(str(partition)) 
f.close()
if not os.path.isdir('data'):
        os.mkdir('data')
nx.write_gexf(G, "./data/test.2024wallstreetbets.gexf", version="1.2draft")


#os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')