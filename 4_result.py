import pandas as pd
import ast
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

res = pd.DataFrame({'d2d':0, '1dTD d2d':0, '5d':0, '10d':0, '20d':0, '1m':0, '2m':0, 'days cnt':0, 'post cnt':0}, index=['total', 'Infl. total'])

def candlestick(ax, start, end): # добавление свечного графика

    data = pd.read_csv('stocks/'+ ticker + '.csv')
    ohlc = pd.DataFrame({'Open':0, 'High':0, 'Low':0, 'Close':0}, index=pd.date_range(start=start, end=end))
    
    data['Date'] = pd.to_datetime(data['Date'])
    ohlc[['Open', 'High', 'Low', 'Close']] = ohlc[['Open', 'High', 'Low', 'Close']].astype(float)
    
    mincoast = 10**9

    for date in pd.date_range(start=start, end=end):
        try:
            mincoast = min(mincoast, data.loc[data['Date'] == date, 'Low'].iloc[0], data.loc[data['Date'] == date, 'Open'].iloc[0], data.loc[data['Date'] == date, 'Close'].iloc[0])
            ohlc.at[date, 'Open'] = data.loc[data['Date'] == date, 'Open'].iloc[0]
            ohlc.at[date, 'High'] = data.loc[data['Date'] == date, 'High'].iloc[0]
            ohlc.at[date, 'Low'] = data.loc[data['Date'] == date, 'Low'].iloc[0]
            ohlc.at[date, 'Close'] = data.loc[data['Date'] == date, 'Close'].iloc[0]
        except:
            None

    up = ohlc[ohlc.Close >= ohlc.Open]
    down = ohlc[ohlc.Close < ohlc.Open]

    col1 = 'g'
    col2 = 'r'

    width = 1
    width2 = .1

    ax.bar(up.index, up.Close-up.Open, bottom=up.Open, color=col1) 
    ax.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1) 
    ax.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1) 
    
    ax.bar(down.index, down.Close-down.Open, bottom=down.Open, color=col2) 
    ax.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2) 
    ax.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)
    ax.set_ylim(bottom=mincoast, top=ohlc.High.max())

def daycorr(df, title): # расчёт количества совпадений по дням (первый показатель эффективности)
    findata = pd.read_csv('stocks/'+ ticker + '.csv')
    #findata['Date'] = pd.to_datetime(findata['Date'])
    findata = findata.set_index('Date')
    ndcorr = 0
    tdcorr = 0
    prev = 0
    tdprev = 0
    for index, row in df.iterrows():
        try:
            diff = findata.at[index.strftime('%Y-%m-%d'), 'Close'] - findata.at[index.strftime('%Y-%m-%d'), 'Open']
            if row['Positive'] + prev > row['Negative'] and diff > 0 or row['Negative'] - prev > row['Positive'] and diff < 0:
                ndcorr += 1
            prev = 0
        except:
            prev += row['Positive'] - row['Negative']
        try:
            diff = findata.at[(index + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d'), 'Close'] - findata.at[(index + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d'), 'Open']
            if row['Positive'] + prev > row['Negative'] and diff > 0 or row['Negative'] - prev > row['Positive'] and diff < 0:
                tdcorr += 1
            tdprev = 0
        except:
            tdprev += row['Positive'] - row['Negative']

    if prev != 0:
        lind = index + pd.Timedelta(1, unit="d")
        while True:
            try:
                diff = findata.at[lind.strftime('%Y-%m-%d'), 'Close'] - findata.at[lind.strftime('%Y-%m-%d'), 'Open']
                if row['Positive'] + prev > row['Negative'] and diff > 0 or row['Negative'] - prev > row['Positive'] and diff < 0:
                    ndcorr += 1
                break
            except:
                lind += pd.Timedelta(1, unit="d")
    
    if tdprev != 0:
        lind = index + pd.Timedelta(1, unit="d")
        while True:
            try:
                diff = findata.at[lind.strftime('%Y-%m-%d'), 'Close'] - findata.at[lind.strftime('%Y-%m-%d'), 'Open']
                if row['Positive'] + prev > row['Negative'] and diff > 0 or row['Negative'] - prev > row['Positive'] and diff < 0:
                    tdcorr += 1
                break
            except:
                lind += pd.Timedelta(1, unit="d")
    if title in res.index:
        res.at[title, 'd2d'] += ndcorr
        res.at[title, '1dTD d2d'] += tdcorr
        res.at[title, 'days cnt'] += df.shape[0]
    else:
        res.loc[title] = [ndcorr, tdcorr, 0, 0, 0, 0, 0, df.shape[0], 0]
    print(title, df.shape[0], ndcorr, tdcorr)

def averageprice(findata, date):
    return (findata.at[date, 'Open'] + findata.at[date, 'Close'] + findata.at[date, 'High'] + findata.at[date, 'Low']) / 4

from sklearn.linear_model import LinearRegression

def trendcoef(start, end):  # линейная регрессия на финансовых данных
    findata = pd.read_csv('stocks/'+ ticker + '.csv')
    findata = findata.loc[findata['Date'].between(start, end)]
    findata['Date'] = (pd.to_datetime(findata['Date']) - pd.Timestamp(start)).dt.days
    model = LinearRegression()
    model.fit(findata['Date'].values.reshape(-1, 1), findata['Close'])

    return model.coef_[0]


def longcorr(df, title): # второй показатель эффективности
    #df['created_utc'] = pd.to_datetime(df['created_utc']).dt.date
    for pub in set(df['post id']):
        """
        собрать суммарное настроение по посту, сверить со стоимостью акции через 5, 10, 30 дней
        """
        postdf = df.loc[df['post id'] == pub]
        lastdate = '2022-01-01'
        firstdate = '2025-01-01'
        pos = neg = 0
        for index, row in postdf.iterrows():
            date = row['created_utc'].strftime('%Y-%m-%d')
            sent = ast.literal_eval(row['sentiment'])['label']
            if sent == 'Positive':
                pos += 1
            if sent == 'Negative':
                neg += 1
            if sent != 'Neutral':
                firstdate = min(firstdate, date)
                lastdate = max(lastdate, date)
        
        if pos + neg > 0 and pos != neg:
            findata = pd.read_csv('stocks/'+ ticker + '.csv')
            findata = findata.set_index('Date')
            while firstdate not in findata.index:
                firstdate = (pd.to_datetime(firstdate) - pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            while lastdate not in findata.index:
                lastdate = (pd.to_datetime(lastdate) - pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            avprice = (averageprice(findata, firstdate) + averageprice(findata, lastdate)) / 2

            d5 = (pd.to_datetime(lastdate) + pd.Timedelta(5, unit="d")).strftime('%Y-%m-%d')
            while d5 not in findata.index:
                d5 = (pd.to_datetime(d5) + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            d10 = (pd.to_datetime(lastdate) + pd.Timedelta(10, unit="d")).strftime('%Y-%m-%d')
            while d10 not in findata.index:
                d10 = (pd.to_datetime(d10) + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            d20 = (pd.to_datetime(lastdate) + pd.Timedelta(20, unit="d")).strftime('%Y-%m-%d')
            while d20 not in findata.index:
                d20 = (pd.to_datetime(d20) + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            m1 = (pd.to_datetime(lastdate) + pd.Timedelta(30, unit="d")).strftime('%Y-%m-%d')
            while m1 not in findata.index:
                m1 = (pd.to_datetime(m1) + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            m2 = (pd.to_datetime(lastdate) + pd.Timedelta(60, unit="d")).strftime('%Y-%m-%d')
            
            while m2 not in findata.index and m2 < '2024-04-01':
                m2 = (pd.to_datetime(m2) + pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')
            while m2 not in findata.index:
                m2 = (pd.to_datetime(m2) - pd.Timedelta(1, unit="d")).strftime('%Y-%m-%d')

            d5p = trendcoef(firstdate, d5)
            d10p = trendcoef(firstdate, d10)
            d20p = trendcoef(firstdate, d20)
            m1p = trendcoef(firstdate, m1)
            m2p = trendcoef(firstdate, m2)
            if pos > neg:
                if d5p > 0:
                    res.at[title, '5d'] += 1
                if d10p > 0:
                    res.at[title, '10d'] += 1
                if d20p > 0:
                    res.at[title, '20d'] += 1
                if m1p > 0:
                    res.at[title, '1m'] += 1
                if m2p > 0:
                    res.at[title, '2m'] += 1
            else:
                if 0 > d5p:
                    res.at[title, '5d'] += 1
                if 0 > d10p:
                    res.at[title, '10d'] += 1
                if 0 > d20p:
                    res.at[title, '20d'] += 1
                if 0 > m1p:
                    res.at[title, '1m'] += 1
                if 0 > m2p:
                    res.at[title, '2m'] += 1
            res.at[title, 'post cnt'] += 1
            print('pos:', pos, end=' ')
            print('neg:', neg, end=' ')
            print(title, pub, avprice, d5p, d10p, d20p, m1p, m2p)
            

def loadFinData(): # загрузка финансовых данных
    start = dt.datetime(2022,1,1)
    end = dt.datetime(2024,4,1)
    df = yf.download(ticker, start, end)
    df = df.drop(['Adj Close'], axis=1)
    if not os.path.isdir('stocks/'):
        os.mkdir('stocks/')
    
    df.to_csv('stocks/' + ticker + '.csv')
    print (ticker,'has data stored to csv file\n')

def addlabels(ax, df): # добавление меток на второй график, отображающий процентное соотношение поз. и нег. тональностей
    for index, row in df.iterrows():
        if row['Count'] > 0:
            if row['Positive'] > 0:
                ax.text(index, row['Pos'] + row['Neg'], s=str(round(row['Positive'] / row['Count'] * 100, 2)) + '%', ha='center', color='g')
            
            if row['Negative'] > 0:
                ax.text(index, row['Neg'], s=str(round(row['Negative'] / row['Count'] * 100, 2)) + '%', ha='center', color='r')
            
            ax.text(index, 0.5, s=int(row['Count']), ha='center', color='black')

def getlabel(s):
    return ast.literal_eval(s)['label']

def timeposts(ax, df, start, end): # первый график, отражающий точное время и день публикации, маркируя цветом тональность
    #df['created_utc'] = pd.to_datetime(df['created_utc']).dt.time
    df['sentiment'] = df['sentiment'].apply(getlabel)
    df = df.loc[df['sentiment'] != 'Neutral']
    d = pd.date_range(start=start, end=end)
    t = set(df['created_utc'].dt.strftime('%H:%M:%S'))
    mt = min(t)
    ax.scatter(d, [mt for i in range(len(d))], s=0)
    
    ax.scatter([start.strftime('%Y-%m-%d') for i in range(len(t))], sorted(t), s=0)
    for index, row in df.iterrows():
        sent = row['sentiment']
        day = row['created_utc'].date()
        time = row['created_utc'].strftime('%H:%M:%S')
        if sent == 'Positive':
            ax.scatter(day, time, color='g')
        if sent == 'Negative':
            ax.scatter(day, time, color='r')


def sentDynamics(df, title): # расчёт динамики тональности дискуссии, в совокупности с построением второго графика
    dfc = df.copy()
    df['created_utc'] = pd.to_datetime(df['created_utc']).dt.date
    dayDistanceStart = 5
    dayDistanceEnd = 30
    start = df['created_utc'].min() - pd.Timedelta(dayDistanceStart, unit="d")
    end = df['created_utc'].max() + pd.Timedelta(dayDistanceEnd + 1, unit="d")
    
    sentdata = pd.DataFrame({'Positive':0, 'Negative':0, 'Count':0, 'Pos':0.0, 'Neg':0.0}, index=pd.date_range(start=start, end=end))
    notNeu = 0
    firstdate = end.strftime('%Y-%m-%d')
    lastdate = start.strftime('%Y-%m-%d')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        date = row['created_utc'].strftime('%Y-%m-%d')
        sent = ast.literal_eval(row['sentiment'])['label']
        if sent != 'Neutral':
            sentdata.at[date, sent] += 1
            sentdata.at[date, 'Count'] += 1
            notNeu += 1
            firstdate = min(firstdate, date)
            lastdate = max(lastdate, date)

    if notNeu > 0:
        start = pd.to_datetime(firstdate) - pd.Timedelta(dayDistanceStart, unit="d")
        end = pd.to_datetime(lastdate) + pd.Timedelta(dayDistanceEnd, unit="d")
        inds = sentdata.index.searchsorted(start)
        inde = sentdata.index.searchsorted(end + pd.Timedelta(1, unit="d"))
        sentdata = sentdata.iloc[inds:inde]
        fig, axs = plt.subplots(3)

        sentdata['Pos'].loc[sentdata['Count'] > 0] = sentdata['Positive'].loc[sentdata['Count'] > 0] / sentdata['Count'].loc[sentdata['Count'] > 0]
        sentdata['Neg'].loc[sentdata['Count'] > 0] = sentdata['Negative'].loc[sentdata['Count'] > 0] / sentdata['Count'].loc[sentdata['Count'] > 0]

        axs[1].bar(sentdata.index, sentdata.Pos + sentdata.Neg, color='g')
        axs[1].bar(sentdata.index, sentdata.Neg, color='r')
        
        axs[0].set_title(title)
        addlabels(axs[1], sentdata)
        candlestick(axs[2], start, end)
        timeposts(axs[0], dfc, start - pd.Timedelta(1, unit="d"), end + pd.Timedelta(1, unit="d"))
        #plt.xticks(rotation=30, ha='right')
        axs[2].set_title(ticker + ' candlestick chart')
        daycorr(sentdata.loc[sentdata['Count'] > 0], title)
        longcorr(df, title)
        return True
    else:
        print(title, 'only Neutral publ.')
        return False

data = pd.read_csv('final.2024wallstreetbets.csv')
data['created_utc'] = pd.to_datetime(data['created_utc'])
tickers = dict()
posts = data.loc[data['type'] == 'post']
ticker = 'None'
maxcnt = 0
subreddit = data['subreddit'].iloc[0]

# поиск постов, в которых упомянут единственный тикер
for index, row in tqdm(posts.iterrows(), total=posts.shape[0]):
    postTickers = ast.literal_eval(row['ORG'])
    if len(postTickers) == 1:
        if postTickers[0] not in tickers:
            tickers[postTickers[0]] = set()
        tickers[postTickers[0]].add(row['id'])
        if len(tickers[postTickers[0]]) > maxcnt:
            maxcnt = len(tickers[postTickers[0]])
            ticker = postTickers[0]

print(tickers)

print('Тикер наиболее часто упоминающейся компании:', ticker, f'- {len(tickers[ticker])} публикаций')


Global_Infl = set()
with open('InflTOP.WSB.txt', 'r') as filehandle:
    for line in filehandle:
        curr_place = line[:-1]
        Global_Infl.add(curr_place)

infltickers = dict()

for t in tickers:
    df = data[data['post id'].isin(tickers[t])]
    authors_ = set(df['Author']) & Global_Infl
    for author in authors_:
        if t not in infltickers:
            infltickers[t] = set()
        infltickers[t].add(author)

print('\nТикеры компаний: {глобальные инфлюенсеры, участвующие в обсуждении}')
print(infltickers)

weights = pd.read_csv('weights.WSB.csv')
weights = weights.set_index('author')

InflBtwnnss = set(weights.nlargest(100, 'Betweenness').index)
InflPageRank = set(weights.nlargest(100, 'PageRank').index)

GI = InflBtwnnss & InflPageRank
clusters = ast.literal_eval(open('clusters.txt', 'r').read())
print('Количество кластеров -', len(clusters))
infl = pd.read_csv('influencers.WSB.csv')
infl = infl.set_index('author')

# вывод графиков динамики тональности дискуссии и динамики фондового рынка по каждому тикеру
# не во всех дискуссиях участвовали выделенные инфлюенсеры и в некоторых дискуссиях, у некоторых инфлюенсеров присутствовали только нейтральные публикации
for ticker in tickers:
    df = data[data['post id'].isin(tickers[ticker])]
    authors = set(df['Author'])
    print('\n' + ticker)
    if not os.path.exists('stocks/' + ticker + '.csv'):
        loadFinData()

    for author in (GI) & authors:
        title = author
        print(author)
        sentDynamics(df.loc[df['Author'] == author], title)
        authorcluster = ast.literal_eval(infl.at[author, 'cluster ind'])
        cluster = set()
        for ind in authorcluster:
            cluster |= clusters[ind]
        sentDynamics(df.loc[df['Author'].isin(cluster)], author + ' cluster')
        
    if len((GI) & authors) > 0:
        sentDynamics(df.loc[df['Author'].isin(GI)], 'Infl. total')
        sentDynamics(df, 'total')
    plt.show()

# подсчёт значений показателей и сохранение результата
res['d2d'] = round(res['d2d'] / res['days cnt'], 2)
res['1dTD d2d'] = round(res['1dTD d2d'] / res['days cnt'], 2)
res['5d'] = round(res['5d'] / res['post cnt'], 2)
res['10d'] = round(res['10d'] / res['post cnt'], 2)
res['20d'] = round(res['20d'] / res['post cnt'], 2)
res['1m'] = round(res['1m'] / res['post cnt'], 2)
res['2m'] = round(res['2m'] / res['post cnt'], 2)
res.to_csv('result.' + subreddit + '.csv')
