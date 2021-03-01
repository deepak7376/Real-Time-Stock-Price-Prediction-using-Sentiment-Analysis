import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()


def clean(review):
  whitespace = re.compile(r"\s+")
  web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
  tesla = re.compile(r"(?i)@Tesla(?=\b)")
  user = re.compile(r"(?i)@[a-z0-9_]+")

  review = re.sub('[^a-zA-Z]', ' ', review)
  review = review.lower()
  review = review.split()
  review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)

  # # we then use the sub method to replace anything matching
  # tweet = whitespace.sub(' ', tweet)
  # tweet = web_address.sub('', tweet)
  # tweet = tesla.sub('Tesla', tweet)
  # tweet = user.sub('', tweet)
  # tweet = [wordnet.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
  return review