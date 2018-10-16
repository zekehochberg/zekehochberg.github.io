---
layout: post
title: "Introduction to NLP tools in Python"
date: 2018-10-16
excerpt: "An introduction to tokenizers and vectorizers and their various parameters in scikit-learn and nltk. Learn how to process natural langauge data before fitting a model."
image: "/assets/img/nlp_intro/nlp_intro.jpg" 
tags: [NLP, sklearn, codealong]
---
# Natural Language Processing Tools in scikit-learn

Natural Language Processing (NLP for short) is the field of making computers understand human languages the way they are written and spoken. You may not realize it, but in every single conversation that you have, your brain is doing an incredible amount of work in order to understand what the person who is talking to you is saying. Because we have all been using language our entire lives, we rarely think about how complicated speech can be. When a word has multiple meanings, how do you know which one a person is trying to use? When somebody uses vague language, like saying it or that, how do we know what they are referring to? How can you tell when someone is being sarcastic? All these questions make NLP an incredibly interesting and challenging field.

Thankfully, there are many applications for NLP, not all of which are so difficult to tackle. There are multiple packages and modules that exist in python to help you with your NLP needs. In this tutorial, we will be focusing on interpreting written text using both the [scikit-learn]('http://scikit-learn.org/stable/') package and the [Natural Language Tool Kit (nltk)]('https://www.nltk.org/') package. We will be going over tokenizers and vectorizers and some of their parameters.

---

Before we get started, you may need to install the scikit-learn and nltk packages on your machine. This can be done by running the following commands from your command line interface

```
pip install sklearn
pip install nltk
```

If you are working in a jupyter notebook environment, you can prepend an exclamation point and run the same code from inside the notebook. Now that the packages are installed, let's jump into the content.

## Vectorizers

When working with any kind of machine learning, models almost always require only numeric data. Vectorizers are a tool that we can use to transform text into an array of numbers that our computers and models can understand. In order to do this, vectorizers will look at a *corpus*, or all of the text that we are trying to analyze. Then, for each individual *document*, or piece of text, the vectorizer will transform that document from words into a numerical representation. If you were analyzing reviews from an online platform, the collection of all reviews would be your corpus, and each individual review would be a document.

There are multiple different vectorizers that use multiple different approaches, but for this post we will focus on only count vectorizers

### Count Vectorizers

Count vectorizers do exactly what their name implies; they get a count of how often each word occurs in a given document. To see how this works, lets create a count vectorizer and an example corpus.


```python
# Import Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate count vectorizer object
cvect = CountVectorizer()

# Create an example corpus to parse
corpus = ['This is my first example sentence.', 
          'Are you as excited to see how this sentence is changed as I am?',
          'This is the last sentence in my corpus']
```

For those who have worked in scikit-learn before, vectorizers will use `.fit()`, `.transform()` and `.fit_transform()` just like any scikit-learn transformer.


```python
# Fit the vectorizer to the example corpus
cvect.fit(corpus)

# Vectorize the example corpus and save it to the variable 'vectorized_example'
vectorized_corpus = cvect.transform(corpus)

# Look at the vectorized corpus
vectorized_corpus
```




    <3x19 sparse matrix of type '<class 'numpy.int64'>'
    	with 26 stored elements in Compressed Sparse Row format>



Our output is a sparse matrix, which is a way of minimizing memory use. To get a better view of our results, we can use the code below to show this sparse matrix as a dataframe


```python
import pandas as pd

pd.DataFrame(vectorized_corpus.todense())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Still not super helpful, as we don't know what the numbers for each feature represent. We can find that out from our count vectorizer object by using the `.get_feature_names()` funcion as we see below.


```python
pd.DataFrame(vectorized_corpus.todense(),
             columns = cvect.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>am</th>
      <th>are</th>
      <th>as</th>
      <th>changed</th>
      <th>corpus</th>
      <th>example</th>
      <th>excited</th>
      <th>first</th>
      <th>how</th>
      <th>in</th>
      <th>is</th>
      <th>last</th>
      <th>my</th>
      <th>see</th>
      <th>sentence</th>
      <th>the</th>
      <th>this</th>
      <th>to</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



When interpreting this output, each row represents a document in our corpus, and each column represents how many times a given word occurred in that document. We can see that in the second sentence (row 1), the words "am" and "are" both occurred once, while the word "as" occurred twice. Now that we've transformed our data, we can use this new dataframe as a feature matrix in any kind of model we would like.

---
### Count Vectorizer Parameters

The above example was a rather simple one, but we can customize our count vectorizer to include some more information. One of the most powerful parameters is the `ngram_range`, which allows the vectorizer to look at groups of words rather than just individual words. This parameter takes a tuple of the minimum and maximum number of consecutive words that the vectorizer will look at. The cell below shows the same sentences vectorized using an `ngram_range` of `(1, 2)`.


```python
cvect = CountVectorizer(ngram_range=(1, 2))
cvect.fit(corpus)
transformed_corpus = cvect.transform(corpus)
pd.DataFrame(transformed_corpus.todense(),
             columns = cvect.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>am</th>
      <th>are</th>
      <th>are you</th>
      <th>as</th>
      <th>as am</th>
      <th>as excited</th>
      <th>changed</th>
      <th>changed as</th>
      <th>corpus</th>
      <th>example</th>
      <th>...</th>
      <th>sentence is</th>
      <th>the</th>
      <th>the last</th>
      <th>this</th>
      <th>this is</th>
      <th>this sentence</th>
      <th>to</th>
      <th>to see</th>
      <th>you</th>
      <th>you as</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 42 columns</p>
</div>



Notice in this new vectorized corpus, we have pairs of words like "are you" and "as am" instead of just individual words. You can adjust the `ngram_range` to capture phrases of any desired length instead of just words. Note that adjusting the minimum value for the `ngram_range` will prevent you from tokenizing individual words as seen below


```python
cvect = CountVectorizer(ngram_range=(2, 2))
cvect.fit(corpus)
transformed_corpus = cvect.transform(corpus)
pd.DataFrame(transformed_corpus.todense(),
             columns = cvect.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>are you</th>
      <th>as am</th>
      <th>as excited</th>
      <th>changed as</th>
      <th>example sentence</th>
      <th>excited to</th>
      <th>first example</th>
      <th>how this</th>
      <th>in my</th>
      <th>is changed</th>
      <th>...</th>
      <th>my corpus</th>
      <th>my first</th>
      <th>see how</th>
      <th>sentence in</th>
      <th>sentence is</th>
      <th>the last</th>
      <th>this is</th>
      <th>this sentence</th>
      <th>to see</th>
      <th>you as</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>



Another major parameter you can change is the `token_pattern` or `tokenizer`. Instead of demonstrating this within the vectorizer, the next section will explicitly go through tokenization.

## Tokenizers

Tokenizers are tools that we use to split a long string of text into smaller pieces. In order to create these smaller pieces, we have to tell our tokenizer how to split up our data. One of the most common ways to do so is by using [regular expressions]('https://en.wikipedia.org/wiki/Regular_expression'). For those unfamiliar with regular expressions, you can find a solid introduction [here]('https://www.regular-expressions.info/') and you can practice using [regex101]('https://regex101.com'). If you don't want to add another thing to learn to your plate, the good news is you don't have to! There are several prebuilt tokenizers that have established patterns that you can use as is.

### RegExp Tokenizer

Using regular expressions, you can specify exactly what patterns you want your tokenizer to look for. I've set up three basic examples below


```python
from nltk.tokenize import RegexpTokenizer

# This will create tokens of only alphanumeric characters
word_tokenizer = RegexpTokenizer(r'\w+') 

# This will create tokens of any non-whitespace characters
non_whitespace_tokenizer = RegexpTokenizer(r'\S+') 

# This will create tokens that are only digits
number_tokenizer = RegexpTokenizer(r'\d+') 
```

Let's take a look at tokenizing the following sentence with each of the three tokenizers.


```python
sample_sentence = "Look at th1s awesome sentence!!! It's fr%om 2018."
```


```python
word_tokenizer.tokenize(sample_sentence)
```




    ['Look', 'at', 'th1s', 'awesome', 'sentence', 'It', 's', 'fr', 'om', '2018']



Notice that the word tokenizer looks only at alphanumeric characters. Anytime the tokenizer encounters a non-alphanumeric character, it creates a new token. This makes the word `"It's"` become two separate tokens, `"It"` and `"s"`. This also causes the word `"fr%om"` to be split at the percent symbol into `"fr"` and `"om"`.


```python
non_whitespace_tokenizer.tokenize(sample_sentence)
```




    ['Look', 'at', 'th1s', 'awesome', 'sentence!!!', "It's", 'fr%om', '2018.']



The non-whitespace tokenizer will only break on whitespace characters. This means that `"It's"` and `"fr%om"` remain as complete tokens. This also makes sure that any word that ends in punctuation, like `"sentence!!!"` and `"2018."` have the punctuation included in the token.


```python
number_tokenizer.tokenize(sample_sentence)
```




    ['1', '2018']



The number tokenizer only returns numbers. It returns the 1 in `"th1s"` and the `2018` at the end of the sentence.

Hopefully you see the power that the regular expressions tokenizer can have, as you can customize exactly what tokens you want to pull from each document. However, for more common use cases there are some pre-built tokenizers within the `nltk` library. While there are many available tokenizers, there is one in particular I'd like to highlight

### MWE Tokenizer

The Multi-Word Expression (MWE) tokenizer is a tool that allows you to combine specific multi-word expressions into their own individual tokens. This can be a really powerful tool that can combine a similar function to the `ngram_range` parameter in your vectorizer


```python
# Import tokenizer module and instantiate tokenizer object
from nltk.tokenize import MWETokenizer
mwe = MWETokenizer()
```

One note, **the MWETokenizer must be used on a list of words that have already been tokenized**. You can see why below.


```python
second_sentence = "This sentence is not good compared to the first one"

mwe.tokenize(second_sentence)
```




    ['T',
     'h',
     'i',
     's',
     ' ',
     's',
     'e',
     'n',
     't',
     'e',
     'n',
     'c',
     'e',
     ' ',
     'i',
     's',
     ' ',
     'n',
     'o',
     't',
     ' ',
     'g',
     'o',
     'o',
     'd',
     ' ',
     'c',
     'o',
     'm',
     'p',
     'a',
     'r',
     'e',
     'd',
     ' ',
     't',
     'o',
     ' ',
     't',
     'h',
     'e',
     ' ',
     'f',
     'i',
     'r',
     's',
     't',
     ' ',
     'o',
     'n',
     'e']



When using the mwe tokenizer on a normal string, it will separate out each individual character. You can get around this by passing in a string with the `.split()` method, or by passing in the tokens created by another tokenizer


```python
# Passing in the string with the .split() method
mwe.tokenize(second_sentence.split())
```




    ['This',
     'sentence',
     'is',
     'not',
     'good',
     'compared',
     'to',
     'the',
     'first',
     'one']




```python
# Passing in tokens created by another tokenizer
word_tokens = word_tokenizer.tokenize(second_sentence)
mwe.tokenize(word_tokens)
```




    ['This',
     'sentence',
     'is',
     'not',
     'good',
     'compared',
     'to',
     'the',
     'first',
     'one']



Now let's add some expressions we'd like to see as individual tokens. These expressions are added as tuples, where each value in the tuple is a word in the expression


```python
mwe.add_mwe(('not', 'good'))

mwe.tokenize(second_sentence.split())
```




    ['This', 'sentence', 'is', 'not_good', 'compared', 'to', 'the', 'first', 'one']



Instead of being separate tokens, the words `not` and `good` are now part of the token `not_good`. We can continue to add as many of these tokens as we want, however these tokens are case sensitive.


```python
mwe.add_mwe(('this', 'sentence'))
```


```python
mwe.tokenize(second_sentence.split())
```




    ['This', 'sentence', 'is', 'not_good', 'compared', 'to', 'the', 'first', 'one']



If we change the example sentence to all lowercase, then we will see `this_sentence` as one individual token.


```python
mwe.tokenize(second_sentence.lower().split())
```




    ['this_sentence', 'is', 'not_good', 'compared', 'to', 'the', 'first', 'one']



Now that you're familiar with tokenizers, you can use them within your vectorizer objects using the `tokenizer` parameter. Note that the `tokenizer` parameter must be a callable function, so instead of just passing the tokenizer object, we want to pass in `object.tokenize`.


```python
# Create the new vectorizer with the custom tokenizer
cvect = CountVectorizer(tokenizer = non_whitespace_tokenizer.tokenize)

# Fit and transform the original corpus
new_vectorized_corpus = cvect.fit_transform(corpus)

# Create a dataframe to visualize the results
pd.DataFrame(new_vectorized_corpus.todense(), columns=cvect.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>am?</th>
      <th>are</th>
      <th>as</th>
      <th>changed</th>
      <th>corpus</th>
      <th>example</th>
      <th>excited</th>
      <th>first</th>
      <th>how</th>
      <th>i</th>
      <th>...</th>
      <th>is</th>
      <th>last</th>
      <th>my</th>
      <th>see</th>
      <th>sentence</th>
      <th>sentence.</th>
      <th>the</th>
      <th>this</th>
      <th>to</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>



Notice that the first token `"am?"` includes the question mark that occured after it, and that both `"sentence"` and `"sentence."` are different tokens. This means our custom tokenizer worked!

There you have it! Hopefully you feel ready to use some of the powerful NLP tools from `nltk` and `sklearn` to tackle any challenges in natural language processing that you may face.
