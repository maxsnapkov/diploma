import praw
import pandas as pd
from datetime import datetime
from time import sleep
import requests
import prawcore

"""
Прежде чем начать работать с API, необходимо зарегистрировать свое приложение на Reddit. 
Для этого выполните следующие действия:

1.Войдите в свою учетную запись на reddit.com.
2.Перейдите на страницу приложений: https://www.reddit.com/prefs/apps.
3.Нажмите на кнопку «Create App» или «Create Another App».
4.Заполните форму, выберите тип приложения (script) и нажмите «Create app».

После создания приложения у вас появятся данные, необходимые для доступа к API: client_id, client_secret и redirect_uri.
"""

reddit = praw.Reddit(client_id = 'ваш client_id',
                     client_secret = 'ваш client_secret',
                     username = 'ваш username',
                     password = 'ваш password',
                     user_agent = 'название user_agent')

df = pd.DataFrame()
srlist = ['wallstreetbets']
for sr in srlist:
    subreddit = reddit.subreddit(sr)

    hot = subreddit.top(time_filter = "year", limit=5) # time_filter отвечает за временной интервал, limit указывает на максимальное количество постов
    k = 0
    for submission in hot:
        df = pd.concat([df, pd.DataFrame([{
                'subreddit': submission.subreddit, # название сабреддита
                'Author': submission.author, # ник автора
                'type': 'post', # тип публикации
                'title': submission.title, # заголовок (на реддите он только у постов)
                'selftext': submission.selftext, # текст публикации
                'upvote_ratio': submission.upvote_ratio, # процент апвоутов (лайков) ко всем реакциям (лайки и дизлайки)
                'ups': submission.ups, # кол-во апвоутов
                'downs': submission.downs, # кол-во даунвоутов, на 2024 год из RedditaAPI просмотр данной статистики недоступен (остался как рудимент)
                'score': submission.score, # так как возможность посмотреть на количество даунвоутов убрана, то теперь это значение эквивалентно количеству лайков
                'created_utc': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%dT%H:%M:%SZ'), # дата и время публикации
                'id': submission.id, # id публикации (можно быстро найти желаемый пост если вставить id в ссылку : reddit.com/r/[название сабреддита, на котором опубликована запись, в работе -- "wallstreetbets"]/comments/[id])
                'post id': submission.id, # id публикации первого уровня (полезно для комментариев)
                'Comment id': 0, # id публикации первого уровня (полезно для комментариев)
                'Parent id': 0 # id родительской публикации на 1 уровень выше (полезно для комментариев)
            }])], ignore_index=True)
        k += 1
        print(submission.id, k)
        
        t = 0
        response = None

        while response == None:
            try:
                submission.comments.replace_more(limit=None) # сбор комментариев, limit указывает на максимальную глубину публикации
                response = submission.comments.list()
                break
            except ConnectionError as e:
                print('Connection error occurred', e)
                sleep(10)
                continue
            except requests.Timeout as e:
                print('Timeout error - request took too long', e)
                sleep(10)
                continue
            except requests.RequestException as e:
                print('General error', e)
                sleep(10)
                continue
            except prawcore.exceptions.PrawcoreException as e:
                print('General error', e)
                sleep(10)
                continue
            except KeyboardInterrupt:
                print('The program has been canceled')
            except:
                break
        
        if response == None:
            continue

        for comment in submission.comments.list():
            df = pd.concat([df, pd.DataFrame([{
                    'subreddit': submission.subreddit,
                    'Author': comment.author,
                    'type': 'comment',
                    'title': submission.title,
                    'selftext': comment.body,
                    'upvote_ratio': 0,
                    'ups': comment.ups,
                    'downs': comment.downs,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'id': comment.id,
                    'post id': submission.id,
                    'Comment id': comment.id,
                    'Parent id': comment.parent_id[3:]
                }])], ignore_index=True)
            t += 1
            print(comment.id, t, end=' ')
        
    df.to_csv('2024wallstreetbets.csv')