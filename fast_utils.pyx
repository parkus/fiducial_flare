import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG_t
ctypedef np.float64_t DBL_t
ctypedef np.int32_t INT_t


def rebin(newbins, oldbbins, oldvalues):
    nb = np.array(newbins, dtype=np.double)
    ob = np.array(oldbbins, dtype=np.double)
    ov = np.array(oldvalues, dtype=np.double)
    n = len(nb) - 1

    nv = np.zeros(n, dtype=np.double)

    if np.any(np.diff(nb) <= 0) or np.any(np.diff(ob) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if (nb[0] < ob[0]) or (nb[-1] > ob[-1]):
        raise ValueError('New bins cannot extend beyond old bins.')

    binmap = np.searchsorted(ob, nb, 'left')

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

    te = np.array(tbins, dtype=np.double)
    nbins = len(te) - 1

    flares = np.array([t0, area_box, height_box, area_decay], dtype=np.double)
    y = np.zeros(nbins, dtype=np.double)
    cdef Py_ssize_t i, j
    for j in range(flares.shape[1]):
        t0, A, h, Ad = flares[:,j]

        if t0 > te[-1]:
            continue

        tau = A/h
        amp = h*tau
        t1 = t0 + A/h
        if t0 < te[0]:
            t0 = te[0]
        i = 0

        # advance to first bin containing part of box
        while te[i+1] < t0 and i < nbins:
            i += 1

        if te[i] < t1:
            # if box is completely within bin, spread box over bin
            if te[i+1] > t1:
                y[i] += h * (t1 - t0)/(te[i+1] - te[i])

            # else spread partial box over bin and advance
            else:
                y[i] += h * (te[i+1] - t0)/(te[i+1] - te[i])
                i += 1

                # add box flux to bin whole bins until reach bin containing end of box
                while te[i+1] < t1:
                    y[i] += h
                    i += 1

                # spread end of box across last bin
                y[i] += h * (t1 - te[i])/(te[i+1] - te[i])

            # spread partial start of exponential over bin
            Ia = 1
            Ib = np.exp(-(te[i+1] - t1)/tau)
            y[i] += amp * (Ia - Ib)/(te[i+1] - te[i])
            i += 1
        else:
            Ib = np.exp(-(te[i] - t1)/tau)

        # add exponential to each of the remaining bins
        while i < nbins:
            Ia = Ib
            Ib = np.exp(-(te[i+1]-t1)/tau)
            y[i] += amp * (Ia - Ib) / (te[i+1] - te[i])
            i += 1

    return y


def boxcar_decay2(tbins, tstart, area_box, height_box, area_decay):

    te = np.array(tbins, dtype=np.double)
    nbins = len(te) - 1
    nflares = len(tstart)

    hb = np.array(height_box, dtype=np.double)
    flares = np.array([tstart, hb, area_box/hb, area_decay, area_decay/hb], dtype=np.double)
    y = np.zeros(nbins, dtype=np.double)
    cdef Py_ssize_t i, j
    cdef double value, ta, tb, t0, h, w, Ad, tau
    for i in range(nbins):
        ta = te[i]
        tb = te[i+1]
        value = 0
        for j in range(nflares):
            t0 = flares[0,j]
            if t0 >= tb:
                continue

            w = flares[2,j]
            h = flares[1,j]
            t1 = t0 + w

            # t0 < tb
            # box
            if t1 > ta: # else fully beyond box
                if t0 > ta:
                    if t1 < tb: # box fully within
                        value += h * w * (t1 - t0)/(tb - ta)
                    else: # box starts within
                        value += h * (tb - t0)/(tb - ta)
                else:
                    if t1 < tb: # box ends within
                        value += h * (t1 - ta)/(tb - ta)
                    else: # box fully covers
                        value += h

            # exponential
            if t1 >= tb:
                continue
            else: # t1 < tb
                Ad = flares[3,j]
                tau = flares[4,j]
                if t1 > ta:
                    value += Ad * (1 - np.exp((t1 - tb)/tau))
                else:
                    value += Ad * (np.exp((t1 - ta)/tau) - np.exp((t1 - tb)/tau))
        y[i] = value

    return y