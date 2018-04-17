import numpy as np
cimport numpy as np

LONG = np.int64
DBL = np.float64
INT = np.int32
ctypedef np.int64_t LONG_t
ctypedef np.float64_t DBL_t
ctypedef np.int32_t INT_t


def rebin(np.ndarray[DBL_t] nb, np.ndarray[DBL_t] ob, np.ndarray[DBL_t] ov):
    n = len(nb) - 1

    cdef np.ndarray[DBL_t] nv = np.zeros(n, dtype=DBL)

    if np.any(np.diff(nb) <= 0) or np.any(np.diff(ob) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if (nb[0] < ob[0]) or (nb[-1] > ob[-1]):
        raise ValueError('New bins cannot extend beyond old bins.')

    cdef np.ndarray[LONG_t] binmap = np.searchsorted(ob, nb, 'left')

    cdef size_t k, i, i0, i1

    for k in range(n):
        i0 = binmap[k]
        i1 = binmap[k+1]
        if i0 == i1:
            nv[k] = (nb[k+1] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
        else:
            left, right, mid = 0.0, 0.0, 0.0
            if nb[k] != ob[i0]:
                left = (ob[i0] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
            if nb[k+1] != ob[i1]:
                right = (nb[k+1] - ob[i1-1]) / (ob[i1] - ob[i1-1]) * ov[i1-1]
                i1 -= 1
            mid = 0.0
            for i in range(i0, i1): mid += ov[i]
            nv[k] = left + mid + right

    return nv/np.diff(nb)


def boxcar_decay(tbins, t0, area_box, height_box, area_decay):

    cdef np.ndarray[DBL_t] te = tbins
    nbins = len(tbins) - 1

    cdef np.ndarray[DBL_t] flares = np.array([t0, area_box, height_box, area_decay]).T
    cdef DBL t, A, h, Ad, t1, E1, tau
    cdef np.ndarray[DBL_T] y = np.zeros(nbins)
    cdef size_t i
    for flare in flares:
        t, A, h, Ad = flare
        tau = A/h
        t1 = t + A/h
        if t < tbins[0]:
            t = tbins[0]
        i = 0

        # advance to first bin containing part of box
        while tbins[i+1] < t:
            i += 1

        # if box is completely within bin, spread box over bin
        if tbins[i+1] < t1:
            y[i] += h * (t1 - t)/(tbins[i+1] - tbins[i])

        # else spread partial box over bin and advance
        else:
            y[i] += h * (tbins[i+1] - t)/(tbins[i+1] - tbins[i])
            i += 1

            # add box flux to bin whole bins until reach bin containing end of box
            while tbins[i+1] < t1:
                y[i] += h
                i += 1

            # spread end of box across last bin
            y[i] += h * (t1 - tbins[i])/(tbins[i+1] - tbins[i])

        # spread partial start of exponential to bin
        y[i] += h/tau * (np.exp(-t1/tau) - np.exp(-tbins[i+1]/tau)) / (tbins[i+1] - tbins[i])
        i += 1

        # add exponential to each of the remaining bins
        while i < nbins:
            y[i] += h/tau * (np.exp(-tbins[i]/tau) - np.exp(-tbins[i+1]/tau)) / (tbins[i+1] - tbins[i])
            i += 1

    return y
