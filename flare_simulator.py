from astropy import table, constants as const, units as u
import numpy as np
import copy
from astropy.io import fits

energy_budget_tbl = table.Table.read('relative_energy_budget.ecsv', format='ascii.ecsv')
eb_bins = np.append(energy_budget_tbl['w0'], energy_budget_tbl['w1'][-1])
eb_density = energy_budget_tbl['Edensity']

# Abbbreviations:
# eqd = equivalent duration
# ks = 1000 s (obvious perhaps :), but not a common unit)

BBfrac_hawley03 = 160
proxy_lines = {'o6':'n5', 'al2':'c2', 'mg2':'o1'}
@u.quantity_input(eqd=u.s)
def boxcar_width_function_default(eqd):
    eqd_s = eqd.to('s').value
    return 7.3*eqd_s**0.19 * u.s
flare_defaults = dict(eqd_min = 100.*u.s,
                      eqd_max = 1e6*u.s,
                      ks_rate = 5.5/u.d,
                      cumulative_index = 0.7,
                      boxcar_width_function = boxcar_width_function_default,
                      decay_boxcar_ratio = 1./2.)


def _kw_or_default(kws, keys):
    values = []
    for key in keys:
        if key not in kws or kws[key] is None:
            kws[key] = flare_defaults[key]
        values.append(kws[key])
    return values


def _check_unit(func, var, unit):
    try:
        var.to(unit)
    except AttributeError, u.UnitConversionError:
        raise('Variable {} supplied to the {} must be an astropy.Units.Quantity object with units convertable to {}'
              ''.format(var.__name__, func.__name__, unit))


def power_rv(min, max, cumulative_index, n):
    if any(isinstance(x, u.Quantity) for x in [min, max, cumulative_index]):
        raise ValueError('No astropy Quantity input for this function, please.')
    # I found it easier to just make my own than figure out the numpy power, pareto, etc. random number generators
    a = cumulative_index
    norm = min**-a - max**-a
    # cdf = 1 - ((x**-a - max**-a)/norm)
    x_from_cdf = lambda c: ((1-c)*norm + max**-a)**(-1/a)
    x_uniform = np.random.uniform(size=n)
    return x_from_cdf(x_uniform)


def shot_times(rate, time_span):
    if any(isinstance(x, u.Quantity) for x in [rate, time_span]):
        raise ValueError('No astropy Quantity input for this function, please.')
    # generate wait times from exponential distribution (for poisson stats)
    # attempt drawing 10 std devs more "shots" than the number expected to fill time_span so chances are very low it
    # won't be filled
    avg_wait_time = 1. / rate
    navg = time_span / avg_wait_time
    ndraw = int(navg + 10*np.sqrt(navg))
    wait_times = np.random.exponential(avg_wait_time, size=ndraw)
    tshot = np.cumsum(wait_times)
    if tshot[-1] < time_span:
        return flare_series(rate, time_span)
    else:
        return tshot[tshot < time_span]


def flare_rate(**kws):
    values = _kw_or_default(kws, ['eqd_min', 'eqd_max', 'ks_rate', 'cumulative_index'])
    eqd_min, eqd_max, ks_rate, cumulative_index = values
    [_check_unit(flare_rate, v, 's') for v in [eqd_min, eqd_max, ks_rate]]

    if eqd_min <= 0:
        raise ValueError('Flare rate diverges at eqd_min == 0. Only eqd_min > 0 makes sense.')
    rate = ks_rate * ((eqd_min/u.ks)**-cumulative_index - (eqd_max/u.ks)**-cumulative_index)
    return rate.to('d-1')


def flare_series(time_span, **kws):
    values = _kw_or_default(kws, ['eqd_min', 'eqd_max', 'ks_rate', 'cumulative_index'])
    eqd_min, eqd_max, ks_rate, cumulative_index = values
    [_check_unit(flare_rate, v, 's') for v in [eqd_min, eqd_max, ks_rate]]

    # get the expected flare rate
    rate = flare_rate(**kws)

    # draw flares at that rate
    tunit = time_span.unit
    rate = rate.to(tunit**-1).value
    time_span = time_span.value
    t_flare = shot_times(rate, time_span) * tunit
    n = len(t_flare)

    # draw energies for those flares
    eqd_min, eqd_max = [x.to(tunit).value for x in [eqd_min, eqd_max]]
    eqd = power_rv(eqd_min, eqd_max, cumulative_index, n) * tunit

    return t_flare, eqd


def boxcar_decay(tbins, t0, area_box, width_box, area_decay):
    if any(isinstance(x, u.Quantity) for x in [tbins, t0, area_box, width_box, area_decay]):
        raise ValueError('No astropy Quantity input for this function, please.')

    # make coarse t,y for boxcar function
    tbox = [tbins[0]] if tbins[0] < t0 else []
    tbox.extend([t0, t0 + width_box])
    height_box = area_box/width_box
    ybox = [0, height_box] if tbins[0] < t0 else [height_box]

    # make precise array for decay
    amp_decay = height_box
    tau_decay = area_decay/amp_decay
    t0_decay = t0 + width_box
    tdecay = tbins[tbins > t0_decay]
    tdecay = np.insert(tdecay, 0, t0_decay)
    Idecay = -amp_decay*tau_decay*np.exp(-(tdecay - t0_decay)/tau_decay)
    ydecay = np.diff(Idecay)/np.diff(tdecay)

    # bin to input stipulations
    t = np.concatenate([tbox[:-1], tdecay])
    y = np.concatenate([ybox, ydecay])
    return rebin(tbins, t, y)


def flare_lightcurve(tbins, t0, eqd, **kws):
    """Return a lightcurve for a single flare normalized to quiescent flux."""
    values = _kw_or_default(kws, ['boxcar_width_function', 'decay_boxcar_ratio'])
    boxcar_width_function, decay_boxcar_ratio = values

    boxcar_width = boxcar_width_function(eqd)
    try:
        boxcar_width.to('s')
    except u.UnitConversionError:
        raise ValueError('boxcar_width_function must return an astropy Quantity with units of time.')
    boxcar_area = eqd/(1 + decay_boxcar_ratio)
    decay_area = boxcar_area * decay_boxcar_ratio

    # make units uniform
    tunit = tbins.unit
    tbins, t0, eqd, boxcar_width, boxcar_area, decay_area = [x.to(tunit).value for x in
                                                             [tbins, t0, eqd, boxcar_width, boxcar_area, decay_area]]
    y = boxcar_decay(tbins, t0, boxcar_area, boxcar_width, decay_area)
    return y


def flare_series_lightcurve(tbins, **kws):

    time_span = tbins[-1] - tbins[0]
    tflares, eqds = flare_series(time_span, **kws)


    lightcurves = [flare_lightcurve(tbins, t, e, **kws) for t, e in zip(tflares, eqds)]
    lightcurves.append(np.zeros(len(tbins)-1))
    return np.sum(lightcurves, 0)




def fiducial_flare_cube(wbins, SiIVenergy, timescale):
    pass


def flare_spectrum(wbins, SiIVenergy):
    return rebin(wbins, eb_bins, eb_density)*SiIVenergy


def rebin(bins_new, bins_old, y):
    if np.any(bins_old[1:] <= bins_old[:-1]) or np.any(bins_new[1:] <= bins_new[:-1]):
        raise ValueError('Old and new bin edges must be monotonically increasing.')

    # compute cumulative integral of binned data
    areas = y*np.diff(bins_old)
    Iold = np.sum(areas) # for accuracy check later
    I = np.cumsum(areas)
    I = np.insert(I, 0, 0)

    # compute average value in new bins
    Iedges = np.interp(bins_new, bins_old, I, left=0, right=0)
    y_new = np.diff(Iedges)/np.diff(bins_new)

    # check that integral using new values is close to the one previous
    Inew = np.sum(np.diff(bins_new)*y_new)
    if not np.isclose(Iold, Inew):
        raise ValueError('Inaccurate result, probably because of very fine binning producing accumulated truncation '
                         'errors in the cumulative sums used in this algorithm. Use a more accurate method (e.g. '
                         'github.com/parkus/crebin)')

    return y_new