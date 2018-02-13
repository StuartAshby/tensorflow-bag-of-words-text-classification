# Tensorflow "Bag of Words" (BoW) text classification
This is a project for [Create a LOOP](https://createaloop.org/) kid's coding club members -- or *anyone* really -- to familiarize with basic AI /ML concepts using [Tensorflow](https://www.tensorflow.org/) + Python and "Bag of Words" (BoW) Natural Language Processing (NLP) text classification. 

![Bag of Words](https://i.ytimg.com/vi/OGK9SHt8SWg/maxresdefault.jpg)

## What is Artificial Intelligence (AI)?
Remember 4 words: ```Intelligence demonstrated by machines```.

## What is Machine Learning (ML)?
Remember 4 more words: ```Machines learning without programming```.

## How do machines demonstrate intelligence and learn without being programmed?
We must train them! Let's get started.

## What is the basis for this project?
An overview of Bag of Words (BoW) [Natural Language Processing](https://en.wikipedia.org/wiki/Natural-language_processing) (NLP) [can be found here](https://ongspxm.github.io/blog/2014/12/bag-of-words-natural-language-processing/). Also familiarize with the NLP concepts of [Stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) and [Tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html).

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks. As always, thank you [Wikipedia](https://en.wikipedia.org/wiki/TensorFlow).

In this project we're going to use Python + Tensorflow to create a text classifier that classifies sentences that it has never encountered before to specific labels that we train it for.

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
pip3 install nltk numpy mkl scipy h5py tflearn tensorflow
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

This opens the ```nltk``` GUI. Download *all the things!* 

## Setup for Windows
*This works on Windows. README under development.*

## Setup for Linux
*This works on Linux. README under development.*

## Test Tensorflow install
Let’s do a "Hello World" program in Python that uses the TensorFlow package to make sure it’s installed correctly. Instead of creating a new Python file, we’ll simply feed the code into Python one line at a time.

You might already be in the Python3 terminal. If not, start Python3:
```
python3
```

Now enter each line of code (don’t include the >>>):
```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, Tensorflow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))

```

You should see: “Hello, Tensorflow!” 

Congratulations! Your computer is now running one of the most powerful machine learning tools on the planet, created by the original Google Brain Team. Your car will one day be able to drive itself thanks to computers running Tensorflow. You have taken your first step towards becoming a data scientist!

Exit the Python3 terminal and return to the Virtualenv prompt:
```
>>> quit()
```

If you'd like to exit your Tensorflow virtual environment type ```deactivate``` and to resume run ```cd ~/tensorflow && source ./bin/activate``` -- where the ```~/tensorflow``` dir is the path where you created your Virtualenv. But we're good to go now, so let's start playing with Tensorflow!

## Play with Tensorflow

Before you begin make sure you see the ```(tensorflow)``` preceding your command prompt. That informs you that your virtual Tensorflow session is activated properly. Activate your virtual environment if you need to.

Clone our project repo and CD to the ```text_classification``` dir to run the code:
```
git clone https://github.com/StuartAshby/tensorflow-bag-of-words-text-classification
cd tensorflow-bag-of-words-text-classification/text_classification
```

The data preparation is in the ```data.json``` file. We included Bag of Words (BoW) classifications for 5 categories:

```
time, sorry, greeting, farewell, age
```

In each of these 5 categories we've added sample sentences to ```data.json``` to train the model.

Remember, it has never seen the sentences we'll attempt to classify, one in each of the 5 categories listed above:
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

We can make it more sophisticated, but this is just a fun starter project to showcase some basic AI / ML using Tensorflow's text classification. Some examples of real-world applications for text classification include Chatbots, document parsing and sentiment analysis, to name just a few.

Enjoy!

*Special thanks to [Akshay Pai](https://github.com/akshaypai) who you can also find writing awesome AI / ML on [Source Dexter](https://sourcedexter.com/author/akshayhpai94/), including [this post](https://sourcedexter.com/tensorflow-text-classification-python/) which informed this project.*
