import collections
import operator
import nltk
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.probability import *
import sys

class BayesianModel:
    """Model that predicts whether or not a sentence has a 
       grammatical error based on n-gram prior probabilities."""

    def __init__(self, n: int, threshold: float):
        self.n = n
        self.threshold = threshold
        self.cfdist = ConditionalFreqDist()
        self.wordcounts = collections.defaultdict(int)

    # hardcodes n=2 for now.
    def build_brown_priors(self):
        print("training brown model...")
        count = 0
        for sent in brown.sents():
            count += 1
            prev = '<s>'
            for word in sent:
                word = word.lower()
                self.wordcounts[word] += 1
                self.cfdist[prev].update([word])
                prev = word
        print("Number of sentences in Brown: ", count)

    def build_reuters_priors(self):
        print("training reuters model...")
        count = 0
        prev = '<s>'
        for word in reuters.words():
            if word in ['.', ';', '?']:
                word = '<s>'
                count += 1
            word = word.lower()
            self.wordcounts[word] += 1
            self.cfdist[prev].update([word])
            prev = word
        print("Number of sentences in Reuters: ", count)

    def build_synthetic_priors(self, filename):
        print("training from synthetic data...")
        count = 0
        with open(filename) as f:
            line = f.readline()
            while line:
                prev = '<s>'
                count += 1
                if count > 1e6:
                    break
                for word in line.split():
                    word = word.lower()
                    self.wordcounts[word] += 1
                    self.cfdist[prev].update([word])
                    prev = word
                line = f.readline()
        print("Number of sentences in synthetic dataset: ", count)
        
    def classify(self, sentence: str):
        bigrams = []
        prev = '<s>'
        for word in sentence.split():
            if word in ['.', ';', '?']:
                continue
            word = word.lower()
            bigrams.append((prev, word))
            prev = word
        score = 0
        contains_zero = False
        for bigram in bigrams:
            (prev, word) = bigram
            mle = MLEProbDist(self.cfdist[prev])
            bigram_score = mle.prob(word)
           # print("counts for ", prev, ": ", self.wordcounts[prev])
           # print("counts for ", word, ": ", self.wordcounts[word])            
           # print("bigram_score for " + str(bigram) + ": " + str(bigram_score))
            score += bigram_score
            if bigram_score <= self.threshold:
                contains_zero = True
        return score, contains_zero

    # Should be in format sentence,label
    # Returns precision for pos and neg classes.
    def classifyFile(self, filename, outputfilename):
        print("classifying file..")
        tp, tn, fp, fn = 0, 0, 0, 0
        with open(filename) as f:
            with open(outputfilename, 'w+') as out:
                line = f.readline()
                while line:
                    label = int(line.split(',')[-1])
                    sentence = line[:-2]
                    score, contains_zero = self.classify(sentence)
                    #pred = score < self.threshold
                    pred = int(contains_zero)
                    if label == 1:
                        if pred == 1: tp += 1
                        if pred == 0: fn += 1
                    if label == 0:
                        if pred == 0: tn += 1
                        if pred == 1: fp += 1
                    out.write(str(label) + "," + str(score) + "\n")
                    line = f.readline()
        print("TP: ", tp)
        print("FN: ", fn)
        print("FP: ", fp)
        print("TN: ", tn)
        return tp / (tp + fn), tn / (tn + fp)

if __name__ == '__main__':
    model = BayesianModel(2, 1e-7)
    model.build_brown_priors()
    model.build_reuters_priors()
    model.build_synthetic_priors(sys.argv[1])
    if len(sys.argv) > 3:
        filename = sys.argv[2]
        outfile = sys.argv[3]
        p, n = model.classifyFile(filename, outfile)
        print("Positive class precision: ", p)
        print("Negative class precision: ", n)
    else:  # Try out sentences.
        while True:
            sentence = input("Enter sentence: ")
            score, contains_zero = model.classify(sentence)
            print("Score is: ", score)
            print("Has grammar error: ", contains_zero)
        
    
    
