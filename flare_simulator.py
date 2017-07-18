from astropy import table, constants as const, units as u
import numpy as np
import os

_mfb_path = os.path.join(os.path.dirname(__file__), 'relative_energy_budget.ecsv')
muscles_flare_budget = table.Table.read(_mfb_path, format='ascii.ecsv')
muscles_flare_budget = muscles_flare_budget.filled(0)
_mfb = muscles_flare_budget
_eb_bins = np.append(_mfb['w0'], _mfb['w1'][-1]) * _mfb['w0'].unit
_eb_density = _mfb['Edensity'].quantity
fuv = [912., 1700.] * u.AA
nuv = [1700., 3200.] * u.AA

# Abbbreviations:
# eqd = equivalent duration
# ks = 1000 s (obvious perhaps :), but not a common unit)


@u.quantity_input(eqd=u.s)
def boxcar_width_function_default(eqd):
    eqd_s = eqd.to('s').value
    return 7.3*eqd_s**0.19 * u.s
flare_defaults = dict(eqd_min = 100.*u.s,
                      eqd_max = 1e6*u.s,
                      ks_rate = 5.5/u.d,
                      cumulative_index = 0.7,
                      boxcar_width_function = boxcar_width_function_default,
                      decay_boxcar_ratio = 1./2.,
                      BB_SiIV_Eratio=160, # Hawley et al. 2003
                      T_BB = 9000*u.K, # Hawley et al. 2003
                      SiIV_quiescent=0.1*u.Unit('erg s-1 cm-2')) # for GJ 832 with bolometric flux equal to Earth
# insolation


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


def flare_lightcurve(tbins, t0, eqd, **flare_params):
    """Return a lightcurve for a single flare normalized to quiescent flux."""
    values = _kw_or_default(flare_params, ['boxcar_width_function', 'decay_boxcar_ratio'])
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


def flare_rate(**flare_params):
    values = _kw_or_default(flare_params, ['eqd_min', 'eqd_max', 'ks_rate', 'cumulative_index'])
    eqd_min, eqd_max, ks_rate, cumulative_index = values
    [_check_unit(flare_rate, v, 's') for v in [eqd_min, eqd_max, ks_rate]]

    if eqd_min <= 0:
        raise ValueError('Flare rate diverges at eqd_min == 0. Only eqd_min > 0 makes sense.')
    rate = ks_rate * ((eqd_min/u.ks)**-cumulative_index - (eqd_max/u.ks)**-cumulative_index)
    return rate.to('d-1')


def flare_series(time_span, **flare_params):
    values = _kw_or_default(flare_params, ['eqd_min', 'eqd_max', 'ks_rate', 'cumulative_index'])
    eqd_min, eqd_max, ks_rate, cumulative_index = values
    [_check_unit(flare_rate, v, 's') for v in [eqd_min, eqd_max, ks_rate]]

    # get the expected flare rate
    rate = flare_rate(**flare_params)

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


def flare_series_lightcurve(tbins, **flare_params):

    time_span = tbins[-1] - tbins[0]
    tflares, eqds = flare_series(time_span, **flare_params)


    lightcurves = [flare_lightcurve(tbins, t, e, **flare_params) for t, e in zip(tflares, eqds)]
    lightcurves.append(np.zeros(len(tbins)-1))
    return np.sum(lightcurves, 0)


def flare_spectrum(wbins, SiIV, **flare_params):
    BBratio, T = _kw_or_default(flare_params, ['BB_SiIV_Eratio', 'T_BB'])

    # get energy density from muscles data
    eb_bins = _eb_bins.to(wbins.unit)
    FUV_and_lines = rebin(wbins.value, eb_bins.value, _eb_density.value) * _eb_density.unit * SiIV

    # get a blackbody and normalize to Hawley value
    red = (wbins[1:] > fuv[1])
    BBbins = np.insert(wbins[1:][red], 0, fuv[1])
    BB = blackbody(BBbins, T, bolometric=BBratio * SiIV)

    result = FUV_and_lines
    result[red] += BB
    return result


@u.quantity_input(wbins=u.AA, T=u.K)
def blackbody(wbins, T, bolometric=None):
    w = (wbins[:-1] + wbins[1:])/2.
    f = np.pi * 2 * const.h * const.c ** 2 / w ** 5 / (np.exp(const.h * const.c / const.k_B / T / w) - 1)
    if bolometric is None:
        return f.to('erg s-1 cm-2 AA-1')
    else:
        fbolo = const.sigma_sb*T**4
        fnorm = (f/fbolo).to(1/wbins.unit)
        return fnorm*bolometric


def flare_series_spectra(wbins, tbins, **flare_params):
    SiIVq = _kw_or_default(flare_params, ['SiIV_quiescent'])
    lightcurve = flare_series_lightcurve(tbins, **flare_params)
    spectrum = flare_spectrum(wbins, SiIVq)
    return np.outer(lightcurve, spectrum)


def rebin(bins_new, bins_old, y):
    if any(isinstance(x, u.Quantity) for x in [bins_new, bins_old, y]):
        raise ValueError('No astropy Quantity input for this function, please.')
    if np.any(bins_old[1:] <= bins_old[:-1]) or np.any(bins_new[1:] <= bins_new[:-1]):
        raise ValueError('Old and new bin edges must be monotonically increasing.')

    # compute cumulative integral of binned data
    areas = y*np.diff(bins_old)
    I = np.cumsum(areas)
    I = np.insert(I, 0, 0)

    # compute average value in new bins
    Iedges = np.interp(bins_new, bins_old, I)
    y_new = np.diff(Iedges)/np.diff(bins_new)

    return y_new