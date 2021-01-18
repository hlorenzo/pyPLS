import numpy as np

def simulateData(n, ncp, p, sigma, signalToNoise=100, ConcVarFact=0.8, n_min_peaks=3):
    """
    Provides a simulated data set X,S,Y
    """

    meanPeakSigma = sigma
    sigPeakSigma = sigma / 4
    axis = np.arange(p)

    S = np.zeros((ncp, p))
    Y = np.zeros((n, ncp))

    for i in np.arange(ncp):
        npeaks = int(np.ceil((np.random.uniform(size=1) * 10)) + 3)
        peakheights = np.random.uniform(size=npeaks)
        sigmas = np.random.uniform(size=npeaks) * sigPeakSigma + meanPeakSigma
        position = np.random.uniform(size=npeaks) * p
        for j in np.arange(npeaks):
            S[i, :] = S[i, :] + peakheights[j] * np.exp(-0.5 * ((axis - position[j]) / sigmas[j]) ** 2)

    meanY = 10 ** np.random.uniform(size=ncp)
    meanY.sort()
    meanY = meanY[::-1]

    varY = ConcVarFact * meanY * np.random.uniform(size=ncp)
    for i in np.arange(ncp):
        Y[:, i] = np.random.normal(loc=meanY[i], scale=varY[i]/2, size=n)

    X = Y.dot(S)
    X = X / np.max(X)
    # Adding some noise

    if signalToNoise:
        Noise = np.random.normal(size=(n, p))
        X = X*signalToNoise + Noise

    return X, S, Y
