#!/usr/bin/python
#
#    This example shows how to use the MITIE Python API to train a named_entity_extractor.
#
#
from time import time
import sys, os
# Make sure you put the mitielib folder into the python search path.  There are
# a lot of ways to do this, here we do it programmatically with the following
# two statements:
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent + '/../../mitielib')

from mitie import *
from sklearn.preprocessing import LabelEncoder


# When you train a named_entity_extractor you need to get a dataset of sentences (or
# sentence or paragraph length chunks of text) where each sentence is annotated with the
# entities you want to find.  For example, if we wanted to find all the names of people and
# organizations then we would need to get a bunch of sentences with examples of person
# names and organizations in them.  Here is an example:
#     My name is Davis King and I work for MIT.
# "Davis King" is a person name and "MIT" is an organization.
#
# You then give MITIE these example sentences with their entity annotations and it will
# learn to detect them.  That is what we do below.

# So let's make the first training example.  We use the sentence above.  Note that the
# training API takes tokenized sentences.  It is up to you how you tokenize them, you
# can use the default tokenizer that comes with MITIE or any other method you like.
intent_sample1 = ["My", "name", "is", "Davis", "King", "and", "I", "work", "for", "MIT", "."]
sample = ner_training_instance(intent_sample1)
# Now that we have the tokens stored, we add the entity annotations.  The first
# annotation indicates that the tokens in the range(3,5) is a person.  I.e.
# "Davis King" is a person name.  Note that you can use any strings as the
# labels.  Here we use "person" and "org" but you could use any labels you
# like.
sample.add_entity(xrange(3,5), "person")
sample.add_entity(xrange(9,10), "org")

# And we add another training example
intent_sample2 = ["The", "other", "day", "at", "work", "I", "saw", "Brian", "Smith", "from", "CMU", "."]
sample2 = ner_training_instance(intent_sample2)
sample2.add_entity(xrange(7,9), "person")
sample2.add_entity(xrange(10,11), "org")


# Now that we have some annotated example sentences we can create the object that does
# the actual training, the ner_trainer.  The constructor for this object takes a string
# that should contain the file name for a saved mitie::total_word_feature_extractor.
# The total_word_feature_extractor is MITIE's primary method for analyzing words and
# is created by the tool in the MITIE/tools/wordrep folder.  The wordrep tool analyzes
# a large document corpus, learns important word statistics, and then outputs a
# total_word_feature_extractor that is knowledgeable about a particular language (e.g.
# English).  MITIE comes with a total_word_feature_extractor for English so that is
# what we use here.  But if you need to make your own you do so using a command line
# statement like:
#    wordrep -e a_folder_containing_only_text_files
# and wordrep will create a total_word_feature_extractor.dat based on the supplied
# text files.  Note that wordrep can take a long time to run or require a lot of RAM
# if a large text dataset is given.  So use a powerful machine and be patient.
trainer = ner_trainer("./data/total_word_feature_extractor_zh.dat")
# Don't forget to add the training data.  Here we have only two examples, but for real
# uses you need to have thousands.
trainer.add(sample)
trainer.add(sample2)
# The trainer can take advantage of a multi-core CPU.  So set the number of threads
# equal to the number of processing cores for maximum training speed.
trainer.num_threads = 4

# This function does the work of training.  Note that it can take a long time to run
# when using larger training datasets.  So be patient.
ner = trainer.train()

# Now that training is done we can save the ner object to disk like so.  This will
# allow you to load the model back in using a statement like:
#   ner = named_entity_extractor("new_ner_model.dat").
#ner.save_to_disk("new_ner_model.dat")

# But now let's try out the ner object.  It was only trained on a small dataset but it
# has still learned a little.  So let's give it a whirl.  But first, print a list of
# possible tags.  In this case, it is just "person" and "org".
print ("tags:", ner.get_possible_ner_tags())


# Now let's make up a test sentence and ask the ner object to find the entities.
tokens = ["I", "met", "with", "John", "Becker", "at", "HBU", "."]
entities = ner.extract_entities(tokens)

feature_extractor = total_word_feature_extractor("./data/total_word_feature_extractor_zh.dat")

# Happily, it found the correct answers, "John Becker" and "HBU" in this case which we
# print out below.
print ("\nEntities found:", entities)
print ("\nNumber of entities detected:", len(entities))

import numpy as np
vector = np.zeros(feature_extractor.num_dimensions)
print ("\nlen(vector):", len(vector))
for e in entities:
    range = e[0]
    tag = e[1]
    entity_text = " ".join(tokens[i] for i in range)
    print ("    " + tag + ": " + entity_text)

for token in tokens:
    _v = feature_extractor.get_feature_vector(token)
    #print ("\nlen(_v):", len(_v))
    #print _v
    vector += _v


print ("\nlen(tokens):", len(tokens))
print ("\nlen(vector):", len(vector))
#print vector
print "####################"
#print vector/len(tokens)
labels = ["info", "ack", "info", "ack", "info", "ack"]
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(labels)
aList = []
intent_samples = [intent_sample1, intent_sample2,intent_sample1,intent_sample2, intent_sample1, intent_sample2]
for intent_sample in intent_samples:
    _vector = np.zeros(feature_extractor.num_dimensions)
    for token in intent_sample:
        _v = feature_extractor.get_feature_vector(token)
        _vector += _v

    aList.append(_vector)


print ("X:", len(aList))
X = np.stack([v for v in aList])
sklearn_config = {
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": "linear"
    }
C = sklearn_config.get("C", [1, 2, 5, 10, 20, 100])
kernel = sklearn_config.get("kernel", "linear")
# dirty str fix because sklearn is expecting str not instance of basestr...
tuned_parameters = [{"C": C, "kernel": [str(kernel)]}]
cv_splits = max(2, min(5, np.min(np.bincount(y)) // 5))  # aim for 5 examples in each fold
print ("cv_splits:", cv_splits)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
t0 = time()
classifier = GridSearchCV(SVC(C=1, probability=True, class_weight='balanced'),
                                    param_grid=tuned_parameters, n_jobs=1,
                                    cv=2, scoring='f1_weighted', verbose=1)

cls = classifier.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()
# best score
print("Best score: %0.3f" % classifier.best_score_)
print("Best parameters set:")
best_parameters = classifier.best_estimator_.get_params()
print (best_parameters)

print classifier.cv_results_

