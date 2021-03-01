# Real-Time-Stock-Price-Prediction-using-Sentiment-Analysis
Use natural-language processing (NLP) to predict stock price movement based on Twitter data and historical stock data.

  
## Website

You are welcome to visit our website: [StockAI.me](http://stockAI.me/). The main purpose of this project is to build the connection between Natural language processing and stock price prediction based on tweets. 

## Methodology

1. Data Collection and Preprocessing

    1.1 Collect the latest tweet regarding the company stock

    1.2 Fetch the tweets using twitter API
    
    1.3 Collect the historical stock data from Yahoo Finance API

2. Feature Engineering (Tokenization)
  
    2.1 Unify word format: unify tense, singular & plural, remove punctuations & stop words
  
3. Using a pre-trained sentiment analysis model from the flair library
4. Use the FBprophet model to predict the stock value
5. Combine sentiment analysis and predictive model to generate final result

## Requirement
* Python 3
* [PyTorch > 0.4](https://pytorch.org/)
* numpy
* [NLTK](https://www.nltk.org/install.html)
* Crawler tools
  - pip3 install yfinance
  - pip3 install requests

## Usage

Note: If you don't want to take time to crawl data and train the model, you can also directly go to step 4.

### 1. Data collection


#### 1.1 Download the ticker list from [NASDAQ](http://www.nasdaq.com/screening/companies-by-industry.aspx)

#### 1.2 Use twitter API to crawl tweets from [twitter]()

*Note: you may need over one month to fetch the tweets you want.*

![](./imgs/baidu.PNG)

We can use the following script to crawl it and format it to our local file

```bash
$ request.py 
```

![](./imgs/111.png)

By brute-force iterating company tickers and dates, we can get the dataset with roughly 1000 tweets in the end. Since a company may have multiple tweets in a single day, the current version will only use top tweet to train our models and ignore the others.

#### 1.3 Use Yahoo finance to crawl historical stock prices
 
Improvement here, use normalized return [5] over S&P 500 instead of return.

```bash
$ stock_data.py # generate raw data: stockPrices_raw.json, containing open, close, ..., adjClose
```

### 2. Feature engineering (Tokenization)

Unify the word format, project word to a word vector, so every sentence results in a matrix.

Detail about unifying word format are: lower case, remove punctuation, get rid of stop words, unify tense and singular & plural.

```bash
$ ./clean.py
```

### 3. Train FBprophet to predict the stock price movement. 
```bash
$ ./forecast.py
```

### 4. Prediction and analysis

Combine the both Prediction. 



### 5. Future works

This is a very rough work. A better label should be based on the comparison of stock price changes between the company and the corresponding industry, instead of the S&P 500, which is in spririt similar to hedging.

By [Tim Loughran and Bill McDonald](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1331573), some words have strong indications of positive and negative effects in finance, e.g. company merger and acquisition. Therefore we need to dig into these words to find more information. In addition, detailed analysis and comparison in each industry are also useful.

Another simple but interesting example can be found in [Financial Sentiment Analysis part1](http://francescopochetti.com/scrapying-around-web/), [part2](http://francescopochetti.com/financial-blogs-sentiment-analysis-part-crawling-web/). 

Since a comprehensive stopword list is quite helpful in improving the prediction power, you are very welcome to build a better stopword list and share it.


## References:

1. Yoon Kim, [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181), EMNLP, 2014
2. J Pennington, R Socher, CD Manning, [GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf), EMNLP, 2014
3. Max Welling, Yee Whye Teh, [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://pdfs.semanticscholar.org/aeed/631d6a84100b5e9a021ec1914095c66de415.pdf), ICML, 2011
4. Tim Loughran and Bill McDonald, 2011, “When is a Liability not a Liability?  Textual Analysis, Dictionaries, and 10-Ks,” Journal of Finance, 66:1, 35-65.
5. H Lee, etc, [On the Importance of Text Analysis for Stock Price Prediction](http://nlp.stanford.edu/pubs/lrec2014-stock.pdf), LREC, 2014
6. Xiao Ding, [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/Proceedings/15/Papers/329.pdf), IJCAI2015
7. [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
8. [Keras predict sentiment-movie-reviews using deep learning](http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)
9. [Keras sequence-classification-lstm-recurrent-neural-networks](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
10. [tf-idf + t-sne](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py)
11. [Implementation of CNN in sequence classification](https://github.com/dennybritz/cnn-text-classification-tf)
12. [Getting Started with Word2Vec and GloVe in Python](http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python)
13. [PyTorch Implementation of Kim's Convolutional Neural Networks for Sentence Classification](https://github.com/Shawn1993/cnn-text-classification-pytorch)
