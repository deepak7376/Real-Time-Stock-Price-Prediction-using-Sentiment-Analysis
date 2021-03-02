import requests

BEARER_TOKEN ="aaaaaa"

params = {
    'q': 'tesla',
    'tweet_mode': 'extended',
    'lang': 'en',
    'count': '100'
}


response = requests.get(
    'https://api.twitter.com/1.1/search/tweets.json',
    params=params,
    headers={'authorization': 'Bearer '+BEARER_TOKEN}
)