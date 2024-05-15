import spacy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import re
import torch
from tqdm import tqdm

def get_split(text1):
  l_total = []
  l_parcial = []
  if len(text1.split())//80 >0:
    n = len(text1.split())//80
  else: 
    n = 1
  for w in range(n):
    if w == 0:
      l_parcial = text1.split()[:100]
      l_total.append(" ".join(l_parcial))
    else:
      l_parcial = text1.split()[w*80:w*80 + 100]
      l_total.append(" ".join(l_parcial))
  return l_total

# чтение баз данных котирующихся акций с бирж Nasdaq и NYSE (данные взяты с открытых источников)
nasdaq = pd.read_csv('nasdaq.csv')
nyse = pd.read_csv('NYSE.csv')

# загрузка модели FinBERT
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fb = pipeline('sentiment-analysis', model=finbert, tokenizer=tokenizer)

# загрузка spaCy
nlp = spacy.load('en_core_web_sm')

data = pd.read_csv('2024wallstreetbets.csv', index_col=0)
data['ORG'] ='None'
data['title sentiment'] = 'None'
data['sentiment'] = 'None'

# создание списка для всех упомянутых в датасете тикеров
symbol = []

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
#    if row[3] == 'post':
        if not isinstance(row['selftext'], str):
            row['selftext'] = ''
        sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', row['selftext'])
        post = nlp(sentence)
        try:
            sentiment = fb(sentence) # может возникнуть проблема обработки больших текстов, далее костыльное решение)
        except:
            try:
                spos = 0
                kpos = 0
                sneu = 0
                kneu = 0
                sneg = 0
                kneg = 0
                temp = get_split(sentence)
                for i in range(len(temp)):
                    stemp = fb(temp[i])
                    if stemp[0]['label'] == 'Negative':
                        sneg += stemp[0]['score']
                        kneg += 1
                    elif stemp[0]['label'] == 'Neutral':
                        sneu += stemp[0]['score']
                        kneu += 1
                    elif stemp[0]['label'] == 'Positive':
                        spos += stemp[0]['score']
                        kpos += 1
                if kpos != 0:
                    spos /= kpos
                if kneg != 0:
                    sneg /= kneg
                if kneu != 0:
                    sneu /= kneu
                if spos > sneg and spos > sneu:
                    sentiment = [{'label': 'Positive', 'score': spos}]
                elif sneg > spos and sneg > sneu:
                    sentiment = [{'label': 'Negative', 'score': sneg}]
                else:
                    sentiment = [{'label': 'Neutral', 'score': sneu}]
            except:
                tokens = tokenizer.encode_plus(sentence, add_special_tokens=False, return_tensors='pt')
                
                input_id_chunks = list(tokens['input_ids'][0].split(510))
                mask_chunks = list(tokens['attention_mask'][0].split(510))
                
                for i in range(len(input_id_chunks)):
                    input_id_chunks[i] = torch.cat([
                    torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])
                    ])
                    mask_chunks[i] = torch.cat([
                    torch.Tensor([i]), mask_chunks[i], torch.Tensor([i])
                    ])

                    pad_len = 512 - input_id_chunks[i].shape[0]
                    # check if tensor length satisfies required chunk size
                    if pad_len > 0:
                        # if padding length is more than 0, we must add padding
                        input_id_chunks[i] = torch.cat([
                            input_id_chunks[i], torch.Tensor([0] * pad_len)
                        ])
                        mask_chunks[i] = torch.cat([
                            mask_chunks[i], torch.Tensor([0] * pad_len)
                        ])


                input_ids = torch.stack(input_id_chunks)
                attention_mask = torch.stack(mask_chunks)

                input_dict = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.int()
                }

                outputs = finbert(**input_dict)

                probs = torch.nn.functional.softmax(outputs[0], dim=-1)
                mean = probs.mean(dim=0)
                m = torch.argmax(mean).item()
                temp = mean.tolist()
                sentiment = [{'label':['Positive', 'Negative', 'Neutral'][m], 'score':temp[m]}]

        org = {X.text for X in post.ents if X.label_ == 'ORG'}

        tickers = []
        temp = list(org)

        # вывод тикеров из названий компаний
        for i in list(temp):
            i = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', i)
            i = re.sub("[^A-Za-z.' ]", "", i)
            if i == '' or i == ' ' or i == 'Reuters':
                continue
            
            ticker = nasdaq.loc[(nasdaq['Symbol'] == i.upper())]
            if ticker.shape[0] != 1:
                ticker = nasdaq.loc[nasdaq['Name'].str.contains(fr'\b{i}\b')==True]
            
            if ticker.shape[0] != 1:
                ticker = nyse.loc[(nyse['Symbol'] == i.upper())]
            if ticker.shape[0] != 1:
                ticker = nyse.loc[nyse['Name'].str.contains(fr'\b{i}\b')==True]

            if not ticker.empty  and ticker.shape[0] == 1:
                ticker = ticker['Symbol'].loc[ticker.index[0]]
                if ticker[:3] == 'JPM':
                    ticker = 'JPM'
                if not (ticker in tickers):
                    tickers.append(ticker)
                if not (ticker in symbol):
                    symbol.append(ticker)
            
        tsentiment = fb(row['title'])
        data.at[index, 'ORG'] = tickers
        data.at[index, 'title sentiment'] = tsentiment[0]
        data.at[index, 'sentiment'] = sentiment[0]

data.to_csv('final.2024wallstreetbets.csv', index=False)

with open("symbol.txt", "w") as output:
    for listitem in symbol:
        output.write(f'{listitem}\n')