import argparse
import json
import re
import sys
from collections import defaultdict


class CoNLLeval:
    """Evaluate the result of processing CoNLL-2000 shared task
    Borrowed from: https://github.com/AdolfVonKleist/rnn-slu/blob/master/rnnslu/CoNLLeval.py
    Evaluate the result of processing CoNLL-2000 shared tasks. This is a
    vanilla python port of the original perl script.
    # usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
    #            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
    # options:   l: generate LaTeX output for tables like in
    #               http://cnts.uia.ac.be/conll2003/ner/example.tex
    #            r: accept raw result tags (without B- and I- prefix;
    #                                       assumes one word per chunk)
    #            d: alternative delimiter tag (default is single space)
    #            o: alternative outside tag (default is O)
    # note:      the file should contain lines with items separated
    #            by $delimiter characters (default space). The final
    #            two items should contain the correct tag and the
    #            guessed tag in that order. Sentences should be
    #            separated from each other by empty lines or lines
    #            with $boundary fields (default -X-).
    # url:       http://lcg-www.uia.ac.be/conll2000/chunking/
    """

    def __init__(self, verbose=0, raw=False, delimiter=" ", otag="O", boundary="-X-"):
        self.verbose = verbose  # verbosity level
        self.boundary = boundary  # sentence boundary
        self.correct = None  # current corpus chunk tag (I,O,B)
        self.correctChunk = 0  # number of correctly identified chunks
        self.correctTags = 0  # number of correct chunk tags
        self.correctType = None  # type of current corpus chunk tag (NP,VP,etc.)
        self.delimiter = delimiter  # field delimiter
        self.FB1 = 0.0  # FB1 score (Van Rijsbergen 1979)
        self.accuracy = 0.0
        self.firstItem = None  # first feature (for sentence boundary checks)
        self.foundCorrect = 0  # number of chunks in corpus
        self.foundGuessed = 0  # number of identified chunks
        self.guessed = None  # current guessed chunk tag
        self.guessedType = None  # type of current guessed chunk tag
        self.i = None  # miscellaneous counter
        self.inCorrect = False  # currently processed chunk is correct until now
        self.lastCorrect = "O"  # previous chunk tag in corpus
        self.latex = 0  # generate LaTeX formatted output
        self.lastCorrectType = ""  # type of previously identified chunk tag
        self.lastGuessed = "O"  # previously identified chunk tag
        self.lastGuessedType = ""  # type of previous chunk tag in corpus
        self.lastType = None  # temporary storage for detecting duplicates
        self.line = None  # line
        self.nbrOfFeatures = -1  # number of features per line
        self.precision = 0.0  # precision score
        self.oTag = otag  # outside tag, default O
        self.raw = raw  # raw input: add B to every token
        self.recall = 0.0  # recall score
        self.tokenCounter = 0  # token counter (ignores sentence breaks)

        self.correctChunk = defaultdict(int)  # number of correctly identified chunks per type
        self.foundCorrect = defaultdict(int)  # number of chunks in corpus per type
        self.foundGuessed = defaultdict(int)  # number of identified chunks per type

        self.features = []  # features on line
        self.sortedTypes = []  # sorted list of chunk type names

    @staticmethod
    def endOfChunk(prevTag, tag, prevType, type, chunkEnd=0):
        """Checks if a chunk ended between the previous and current word.
        Checks if a chunk ended between the previous and current word.
        Args:
            prevTag (str): Previous chunk tag identifier.
            tag (str): Current chunk tag identifier.
            prevType (str): Previous chunk type identifier.
            type (str): Current chunk type identifier.
            chunkEnd (int): 0/True true/false identifier.
        Returns:
            int: 0/True true/false identifier.
        """
        if prevTag == "B" and tag == "B":
            chunkEnd = True
        if prevTag == "B" and tag == "O":
            chunkEnd = True
        if prevTag == "I" and tag == "B":
            chunkEnd = True
        if prevTag == "I" and tag == "O":
            chunkEnd = True
        if prevTag == "E" and tag == "E":
            chunkEnd = True
        if prevTag == "E" and tag == "I":
            chunkEnd = True
        if prevTag == "E" and tag == "O":
            chunkEnd = True
        if prevTag == "I" and tag == "O":
            chunkEnd = True
        if prevTag != "O" and prevTag != "." and prevType != type:
            chunkEnd = True
        # corrected 1998-12-22: these chunks are assumed to have length 1
        if prevTag == "]":
            chunkEnd = True
        if prevTag == "[":
            chunkEnd = True

        return chunkEnd

    @staticmethod
    def startOfChunk(prevTag, tag, prevType, type, chunkStart=0):
        """Checks if a chunk started between the previous and current word.
        Checks if a chunk started between the previous and current word.
        Args:
            prevTag (str): Previous chunk tag identifier.
            tag (str): Current chunk tag identifier.
            prevType (str): Previous chunk type identifier.
            type (str): Current chunk type identifier.
            chunkStart:
        Returns:
            int: 0/True true/false identifier.
        """
        if prevTag == "B" and tag == "B":
            chunkStart = True
        if prevTag == "I" and tag == "B":
            chunkStart = True
        if prevTag == "O" and tag == "B":
            chunkStart = True
        if prevTag == "O" and tag == "I":
            chunkStart = True
        if prevTag == "E" and tag == "E":
            chunkStart = True
        if prevTag == "E" and tag == "I":
            chunkStart = True
        if prevTag == "O" and tag == "E":
            chunkStart = True
        if prevTag == "O" and tag == "I":
            chunkStart = True
        if tag != "O" and tag != "." and prevType != type:
            chunkStart = True
        # corrected 1998-12-22: these chunks are assumed to have length 1
        if tag == "[":
            chunkStart = True
        if tag == "]":
            chunkStart = True
        return chunkStart

    def Evaluate(self, infile):
        """Evaluate test outcome for a CoNLLeval shared task.
        Evaluate test outcome for a CoNLLeval shared task.
        Args:
            infile (str): The input file for evaluation.
        """
        with open(infile, "r") as ifp:
            for line in ifp:
                line = line.lstrip().rstrip()
                self.features = re.split(self.delimiter, line)
                if len(self.features) == 1 and re.match(r"^\s*$", self.features[0]):
                    self.features = []
                if self.nbrOfFeatures < 0:
                    self.nbrOfFeatures = len(self.features) - 1
                elif self.nbrOfFeatures != len(self.features) - 1 and len(self.features) != 0:
                    raise ValueError("Unexpected number of features: {0}\t{1}".format(len(self.features) + 1,
                                                                                      self.nbrOfFeatures + 1))
                if len(self.features) == 0 or self.features[0] == self.boundary:
                    self.features = [self.boundary, "O", "O"]
                if len(self.features) < 2:
                    raise ValueError("CoNLLeval: Unexpected number of features in line.")

                if self.raw is True:
                    if self.features[-1] == self.oTag:
                        self.features[-1] = "O"
                    if self.features[-2] == self.oTag:
                        self.features[-2] = "O"
                    if not self.features[-1] == "O":
                        self.features[-1] = "B-{0}".format(self.features[-1])
                    if not self.features[-2] == "O":
                        self.features[-2] = "B-{0}".format(self.features[-2])
                # 20040126 ET code which allows hyphens in the types
                ffeat = re.search(r"^([^\-]*)-(.*)$", self.features[-1])
                if ffeat:
                    self.guessed = ffeat.groups()[0]
                    self.guessedType = ffeat.groups()[1]
                else:
                    self.guessed = self.features[-1]
                    self.guessedType = ""

                self.features.pop(-1)
                ffeat = re.search(r"^([^\-]*)-(.*)$", self.features[-1])
                if ffeat:
                    self.correct = ffeat.groups()[0]
                    self.correctType = ffeat.groups()[1]
                else:
                    self.correct = self.features[-1]
                    self.correctType = ""
                self.features.pop(-1)

                if self.guessedType is None:
                    self.guessedType = ""
                if self.correctType is None:
                    self.correctType = ""

                self.firstItem = self.features.pop(0)

                # 1999-06-26 sentence breaks should always be counted as out of chunk
                if self.firstItem == self.boundary:
                    self.guessed = "O"

                if self.inCorrect is True:
                    if self.endOfChunk(self.lastCorrect, self.correct, self.lastCorrectType, self.correctType) is True \
                            and self.endOfChunk(self.lastGuessed, self.guessed,
                                                self.lastGuessedType, self.guessedType) is True \
                            and self.lastGuessedType == self.lastCorrectType:
                        self.inCorrect = False
                        self.correctChunk[self.lastCorrectType] += 1
                    elif self.endOfChunk(self.lastCorrect, self.correct, self.lastCorrectType, self.correctType) != \
                            self.endOfChunk(self.lastGuessed, self.guessed, self.lastGuessedType, self.guessedType) or \
                            self.guessedType != self.correctType:
                        self.inCorrect = False

                if self.startOfChunk(self.lastCorrect, self.correct, self.lastCorrectType, self.correctType) is True \
                        and self.startOfChunk(self.lastGuessed, self.guessed,
                                              self.lastGuessedType, self.guessedType) is True \
                        and self.guessedType == self.correctType:
                    self.inCorrect = True

                if self.startOfChunk(self.lastCorrect, self.correct, self.lastCorrectType, self.correctType) is True:
                    self.foundCorrect[self.correctType] += 1

                if self.startOfChunk(self.lastGuessed, self.guessed, self.lastGuessedType, self.guessedType) is True:
                    self.foundGuessed[self.guessedType] += 1

                if self.firstItem != self.boundary:
                    if self.correct == self.guessed and self.guessedType == self.correctType:
                        self.correctTags += 1
                    self.tokenCounter += 1

                self.lastGuessed = self.guessed
                self.lastCorrect = self.correct
                self.lastGuessedType = self.guessedType
                self.lastCorrectType = self.correctType

                if self.verbose > 1:
                    print("{0} {1} {2} {3} {4} {5} {6}".format(self.lastGuessed, self.lastCorrect, self.lastGuessedType,
                                                               self.lastCorrectType, self.tokenCounter,
                                                               len(self.foundCorrect.keys()),
                                                               len(self.foundGuessed.keys())))

        if self.inCorrect is True:
            self.correctChunk[len(self.correctChunk.keys())] = 0
            self.correctChunk[self.lastCorrectType] += 1

    def ComputeAccuracy(self):
        """Compute overall precision, recall and FB1 (default values are 0.0).
        Compute overall precision, recall and FB1 (default values are 0.0).
        Results:
            list: accuracy, precision, recall, FB1 float values.
        """
        if sum(self.foundGuessed.values()) > 0:
            self.precision = 100 * sum(self.correctChunk.values()) / float(sum(self.foundGuessed.values()))
        if sum(self.foundCorrect.values()) > 0:
            self.recall = 100 * sum(self.correctChunk.values()) / float(sum(self.foundCorrect.values()))
        if self.precision + self.recall > 0:
            self.FB1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        overall = "processed {0} tokens with {1} phrases; found: {2} phrases; correct: {3}."
        overall = overall.format(self.tokenCounter, sum(self.foundCorrect.values()), sum(self.foundGuessed.values()),
                                 sum(self.correctChunk.values()))
        if self.verbose > 0:
            print(overall)

        self.accuracy = 100 * self.correctTags / float(self.tokenCounter)
        if self.tokenCounter > 0 and self.verbose > 0:
            print("accuracy:  {0:0.2f}".format(self.accuracy))
            print("precision: {0:0.2f}".format(self.precision))
            print("recall:    {0:0.2f}".format(self.recall))
            print("FB1:       {0:0.2f}".format(self.FB1))

        return {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "FB1": self.FB1}

    def conlleval(self, predictions, groundtruth, words, infile):
        """Evaluate the results of one training iteration.

        Evaluate the results of one training iteration.  This now
        uses the native python port of the CoNLLeval perl script.
        It computes the accuracy, precision, recall and FB1 scores,
        and returns these as a dictionary.
        Args:
            predictions (list): Predictions from the network.
            groundtruth (list): Ground truth for evaluation.
            words (list): Corresponding words for de-referencing.
            infile:
        Returns:
            dict: Accuracy (accuracy), precisions (p), recall (r), and FB1 (f1) scores represented as floats.
            infile: The inputs written to file in the format understood by the conlleval.pl script and CoNLLeval python
                    port.
        """
        ofp = open(infile, "w")
        for sl, sp, sw in zip(groundtruth, predictions, words):
            ofp.write(u"BOS O O\n")
            for wl, wp, words in zip(sl, sp, sw):
                line = u"{0} {1} {2}\n".format(words, wl, wp)
                ofp.write(line)
            ofp.write(u"EOS O O\n\n")
        ofp.close()
        self.Evaluate(infile)
        return self.ComputeAccuracy()


if __name__ == "__main__":

    example = "{0} --infile".format(sys.argv[0])
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--infile", "-i", help="Input CoNLLeval results file.", required=True)
    parser.add_argument("--raw", "-r", help="Accept raw result tags.", default=False, action="store_true")
    parser.add_argument("--delimiter", "-d", help="Token delimiter.", default=" ", type=str)
    parser.add_argument("--otag", "-ot", help="Alternative outside tag.", default="O", type=str)
    parser.add_argument("--boundary", "-b", help="Boundary tag.", default="-X-", type=str)
    parser.add_argument("--verbose", "-v", help="Verbose mode.", default=0, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        for key, val in args.__dict__.iteritems():
            print("{0}:  {1}".format(key, val))

    ce = CoNLLeval(verbose=args.verbose, raw=args.raw, delimiter=args.delimiter, otag=args.otag, boundary=args.boundary)
    ce.Evaluate(args.infile)
    results = ce.ComputeAccuracy()

    print()
    json.dumps(results, indent=4)
