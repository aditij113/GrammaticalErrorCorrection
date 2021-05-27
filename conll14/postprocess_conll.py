import sys

# Parses M2 test data into a file of the format: sentence, {1|0}
# 1 indicates grammar error, and 0 indicates no grammar error.
def postprocess(inputFilename, outputFilename):
    prevSentence = ""
    lastSeen = 'A'
    data = {}
    with open(inputFilename) as f:
        line = f.readline()
        while line:

            line = line.split()
            
            if not line:
                line = f.readline()
                continue
            
            if line[0] == 'S':  # New sentence
                lastSeen = 'S'
                if prevSentence != "":
                    data[prevSentence] = 0  # No grammar error.
                prevSentence = ' '.join(line[1:])
                
            elif line[0] == 'A':  # Sentence has error
                if lastSeen != 'A':  # Ensures we only add one error per sentence.
                    data[prevSentence] = 1  # Grammar error
                    prevSentence = ""
                lastSeen = 'A'
                
            line = f.readline()

    # Print number of positives and negatives.
    print("Num sentences with grammar error: ", sum(x == 1 for x in data.values()))
    print("Num sentences without grammar error: ", sum(x == 0 for x in data.values()))

    # Write out to file in form sentence,{0|1}.
    with open(outputFilename, "w+") as fp:
        for sentence,label in data.items():
            fp.write(sentence + ',' + str(label) + '\n')


if __name__ == '__main__':
    m2Filename = sys.argv[1]
    outputFilename = sys.argv[2]
    print("m2Filename: ", m2Filename)
    print("outputFilename: ", outputFilename)
    postprocess(m2Filename, outputFilename)
