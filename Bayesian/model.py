import operator
import nltk
from nltk.corpus import brown
from nltk.probability import *

class BayesianModel:
    """Model that predicts whether or not a sentence has a 
       grammatical error based on n-gram prior probabilities."""

    def __init__(self, n: int, threshold: int):
        self.n = n
        self.threshold = threshold

    # hardcodes n=2 for now.
    def build_priors(self):
        self.cfdist = ConditionalFreqDist()
        prev = '<s>'
        for word in brown.words():
            self.cfdist[prev].update([word])
        prev = word
        
    def classify(self, sentence: str):
        bigrams = []
        prev = '<s>'
        for word in sentence:
            bigrams.append((prev, word))
            prev = word
        score = 0
        for bigram in bigrams:
            # Conside Laplace: laplace = LaplaceProbDist(cfdist[word], 20000)
            mle = MLEProbDist(self.cfdist[bigram[0]])
            score += mle.prob(bigram[1])
        return score

if __name__ == '__main__':
    model = BayesianModel(0, 0)
    model.build_priors()
    while True:
        sentence = input("Enter sentence: ")
        score = model.classify(sentence)
        print("Score is: ", score)
        
    
    
