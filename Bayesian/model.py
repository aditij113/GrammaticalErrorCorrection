import operator
import nltk
from nltk.corpus import brown
from nltk.probability import *
import sys

class BayesianModel:
    """Model that predicts whether or not a sentence has a 
       grammatical error based on n-gram prior probabilities."""

    def __init__(self, n: int, threshold: float):
        self.n = n
        self.threshold = threshold

    # hardcodes n=2 for now.
    def build_priors(self):
        self.cfdist = ConditionalFreqDist()
        prev = '<s>'
        for word in brown.words():
            word = word.lower()
            if word == '.': word = '<s>'
            self.cfdist[prev].update([word])
            prev = word
        
    def classify(self, sentence: str):
        bigrams = []
        prev = '<s>'
        for word in sentence.split():
            word = word.lower()
            bigrams.append((prev, word))
            prev = word
        score = 0
        for bigram in bigrams:
            # Conside Laplace: laplace = LaplaceProbDist(cfdist[word], 20000)
            mle = MLEProbDist(self.cfdist[bigram[0]])
            score += mle.prob(bigram[1])
        return score < self.threshold

    # Should be in format sentence,label
    # Returns precision for pos and neg classes.
    def classifyFile(self, filename):
        tp, tn, fp, fn = 0, 0, 0, 0
        with open(filename) as f:
            line = f.readline()
            while line:
                label = int(line.split(',')[-1])
                sentence = line[:-2]
                pred = self.classify(sentence)
                if label == 1:
                    if pred == 1: tp += 1
                    if pred == 0: fn += 1
                if label == 0:
                    if pred == 0: tn += 1
                    if pred == 1: fp += 1
                line = f.readline()
        return tp / (tp + fn), tn / (tn + fp)

if __name__ == '__main__':
    print("number of sentences: ", len(brown.sents()))
    print("number of words: ", len(brown.words()))
    model = BayesianModel(2, 0.3)
    model.build_priors()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        p, n = model.classifyFile(filename)
        print("Positive class precision: ", p)
        print("Negative class precision: ", n)
    else:  # Try out sentences.
        while True:
            sentence = input("Enter sentence: ")
            pred  = model.classify(sentence)
            print("Pred is: ", pred)
        
    
    
