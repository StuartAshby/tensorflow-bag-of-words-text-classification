# Tensorflow "Bag of words" Text Classification
"Bag of Words" NLP AI text classification with Python + Tensorflow.

An overview of Bag of Words (BoW) Natural Language Processing (NLP) [can be found here](https://ongspxm.github.io/blog/2014/12/bag-of-words-natural-language-processing/). Also familiarize with the NLP concepts of [Stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) and [Tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html).

In this project we're going to use Python + Tensorflow to build a text classifier that classifies a given sentence to one of the labels that we train the classifier for.

Let's begin. First, let's make sure we set our environment up properly. We'll need Python3 to run this due to some dependencies. Let's get Tensorflow fired up. We'll run this in a virtual environment to keep it separate and interference-free from other Python stuff on our machine.

## Setup for Mac
Install Python 3:
```
brew install python3
```

If you already have a Virtualenv installed, you may need to uninstall it:
```
sudo -H pip3 uninstall virtualenv
```
or if you have Python 2.x:
```
sudo -H pip uninstall virtualenv
```

Now we can install the Virtualenv using pip3:
```
sudo -H pip3 install virtualenv
```

Create a new Virtualenv environment:
```
virtualenv --system-site-packages ~/tensorflow
```

Activate the Virtualenv environment:
```
cd ~/tensorflow
source ./bin/activate
```

This will change your bash prompt to the following:
```
(tensorflow)$
```

Check to make sure you are running Python 3.6.x:
```
python --version
```

This should return ```Python 3.6.4```.

Let's install some Python3 package dependencies:
```
pip3 install nltk

pip3 install numpy

pip3 install mkl

pip3 install scipy

pip3 install h5py

pip3 install tflearn

pip3 install tensorflow
```

And we also need to install some ```nltk``` dependencies:
```
python3
```

This will open the Python3 terminal.
```
>>> import nltk
>>> nltk.download()

```

This opens the ```nltk``` GUI. Download *all the things!* Once the download is complete, you can exit the Python3 terminal and return to the Virtualenv prompt

```
>>> quit()
```

## Play with Tensorflow

Before you begin make sure you see the ```(tensorflow)``` preceding your command prompt. That informs you that your virtual Tensorflow session is activated properly.

Clone our project repo and CD to the ```text_classification``` dir to run the code:
```
git clone https://github.com/StuartAshby/tensorflow-bag-of-words-text-classification
cd tensorflow-bag-of-words-text-classification/text_classification
```

The data preparation is in the ```data.json``` file. We included Bag of Words (BoW) classifications for 5 categories:

```
time
sorry
greeting
farewell
age
```

In each of these 5 categories we've added sample sentences to ```data.json``` to train the model.

But it's never seen the sentences we'll attempt to classify, one in each of the 5 categories listed above:
```
"Do you know the time?" - time
"I apologize for being rude." - sorry
"Hey what's up?" - greeting
"See you later!" - farewell
"You must be a couple of years older than him!" - age
```

For now, we've hard-coded these into the ```classify_text.py``` code, along with the data loading, pre-processing, Tensorflow specification data conversion and the actual text classification piece.

We'll run 1,000 epochs for the purpose of this exercise (```n_epoch=1000```). This amounts to 3,000 training steps. This is a lightweight test. Tensorflow will perform much better on GPUs and can be pushed to higher epochs and training steps.

When we run it, it will attempt to train it and classify these sentences above -- which it has never encountered before -- into the categories we've trained it on:
```
python3 classify_text.py
```

Nice! With only a limited run, Tensorflow has correctly classified all 5 of our out-of-sample, never-before-seen sentences at a ```96% accuracy```.:
```
time
sorry
greeting
farewell
age
```

We can make it more sophisticated, but this is just a fun starter project to showcase the power of AI & ML using Tensorflow's text classification. Some examples of real-world applications for text classification include Chatbots, document parsing and sentiment analysis, to name just a few.

Enjoy!!!
