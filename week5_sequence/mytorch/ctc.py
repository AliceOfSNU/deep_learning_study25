import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        extSymbols = []
        skipConnect = []

        extSymbols.append(self.BLANK)
        for sym in target:
            extSymbols.append(sym)
            extSymbols.append(self.BLANK)
        for i in range(len(extSymbols)):
            if extSymbols[i] != self.BLANK and i-2 >= 0 and extSymbols[i-2] != extSymbols[i]:
                skipConnect.append(1)
            else:
                skipConnect.append(0)


        return np.asarray(extSymbols), np.asarray(skipConnect)

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        alpha[0, 0] = logits[0, extSymbols[0]]
        alpha[0, 1] = logits[0, extSymbols[1]]
        for t in range(1, T):
            alpha[t, 0] = alpha[t-1, 0] * logits[t, extSymbols[0]]
            for s in range(1, S):
                alpha[t, s] = alpha[t-1, s] + alpha[t-1, s-1]
                if skipConnect[s]:
                    alpha[t, s] += alpha[t-1, s-2]
                alpha[t, s] *= logits[t, extSymbols[s]]
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        beta[T-1, S-1] = logits[T-1, extSymbols[S-1]]
        beta[T-1, S-2] = logits[T-1, extSymbols[S-2]]
        for t in range(T-2, -1, -1):
            beta[t, S-1] = beta[t+1, S-1] * logits[t, extSymbols[S-1]]
            for s in range(S-2, -1, -1):
                beta[t, s] = beta[t+1, s] + beta[t+1, s+1]
                if s+2 < S and skipConnect[s+2]:
                    beta[t, s] += beta[t+1, s+2]
                beta[t, s] *= logits[t, extSymbols[s]]

        for t in range(T):
            for s in range(S):
                beta[t, s] /= logits[t, extSymbols[s]]
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """

        # -------------------------------------------->

        # Your Code goes here
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1)[:,None]

        # <---------------------------------------------

        return gamma
