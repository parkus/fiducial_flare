from astropy import table, constants as const, units as u
import numpy as np
import copy
from astropy.io import fits

energy_budget_tbl = table.Table.read('relative_energy_budget.ecsv', format='ascii.ecsv')
eb_bins = np.append(energy_budget_tbl['w0'], energy_budget_tbl['w1'][-1])
eb_density = energy_budget_tbl['Edensity']
eb_spec = Spectrum(eb_bins, eb_density)

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
star_distances = {'gj1214': 14.55,
                 'gj176': 9.27,
                 'gj436': 10.13,
                 'gj581': 6.21,
                 'gj667c': 6.8,
                 'gj832': 4.95,
                 'gj876': 4.69,
                 'hd103095': 9.092,
                 'hd40307': 13.0,
                 'hd85512': 11.16,
                 'hd97658': 21.11,
                 'v-eps-eri': 3.22,
                 'gj551': 1.302}
_bandwidth = 100*u.km/u.s
_line_centers_SiIV = [1393.76, 1402.77]
band_SiIV = []


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


def fiducial_flare_spectrum(wbins, SiIVenergy):
    spec = eb_spec.rebin(wbins)
    spec.y *= SiIVenergy
    return spec


def read_MUSCLES_spectrum(path, instellation=None, distance=None):
    if not (instellation is None or distance in None):
        raise ValueError('Can specify an instellation or a distance, but not both.')

    spec = table.Table.read(path)
    wbins = np.append(spec['WAVELENGTH0'], spec['WAVELENGTH1'][-1]) * spec['WAVELENGTH0'].unit

    # get flux, scaling as desired
    if instellation is None and distance is None:
        flux = spec['FLUX'].quantity
    elif instellation is not None:
        bolo = spec['BOLOFLUX']
        flux = bolo.data * u.Unit(bolo.unit.to_string()) * instellation # units clooged because of some bug making
        # boloflux column show up with unrecognized unit
        flux = flux.to('erg s-1 cm-2 AA-1')
    elif distance is not None:
        star = fits.getval(path, 'targname')
        dstar = star_distances[star.lower()] * u.pc
        scale = (dstar/distance)**2
        scale = scale.to('').value
        flux = spec['FLUX'].quantity*scale

    spec = Spectrum(wbins, flux, yname=['flux, f'], notes=['MUSCLES spectrum read from {}'.format(path)])
    return spec


def fiducial_flare_cube(wbins, SiIVenergy, timescale):
    pass


class Spectrum(object):
    file_suffix = '.spec'
    table_write_format = 'ascii.ecsv'

    # these prevent recursion with __getattr__, __setattr__ when object is being initialized
    ynames = []
    other_data = {}

    @u.quantity_input(wbins='length')
    def __init__(self, wbins, y, err=None, yname='y', notes=None):
        """

        Parameters
        ----------
        wbins : astropy Quantity array
            Wavelength bin edges.
        y : astropy Quantity array or numpy array
            y data (e.g. flux, cross-section). If not a Quantity, assumed unitless.
        err : astropy Quantity array or numpy array
            error on y data. must have units equivalent to y
        yname : str | list
            name for the primary dependent data ('flux', 'x', ...). data will be accessible as an attribute such as 
            spec.flux or spec.f if yname=['flux', 'f'].
        notes : str | list
        """
        # vet input
        if not isinstance(y, u.Quantity):
            y = y*u.Unit('')
        if err is not None:
            if not isinstance(err, u.Quantity):
                err = err*u.Unit('')
            try:
                err.to(y)
            except u.UnitConversionError:
                raise 'Units of err must be consistent with units of y.'

        self.wbins = wbins
        self.y = y
        self.e = self.err = self.error = err
        self.ynames = yname if type(yname) is list else [yname]
        self.notes = notes if type(notes) is list else [notes]

    dw = property(lambda self: np.diff(self.wbins))
    w = property(lambda self: midpts(self.wbins))
    integral = property(lambda self: np.sum(self.dw * self.y).decompose())

    def __getattr__(self, key):
        if key in self.ynames:
            return self.y
        else:
            raise AttributeError('No {} attribute. Consider specifying as a yname when initializing spectrum.'
                                 ''.format(key))

    def __setattr__(self, key, value):
        if key in self.ynames:
            self.__dict__['y'] = value
        else:
            self.__dict__[key] = value

    def to_table(self):
        data = [self.w, self.dw, self.y]
        names = ['w', 'dw', 'y']
        if self.e is not None:
            data.append(self.e)
            names.append('err')
        tbl = table.Table(data=data, names=names)
        tbl.meta['ynames'] = self.ynames
        tbl.meta['notes'] = self.notes
        return tbl

    def __repr__(self):
        tbl = self.to_table()
        return tbl.__repr__()

    def __len__(self):
        return len(self.y)

    def add_note(self, note):
        self.notes.append(note)

    def rebin(self, newbins, other_data_bin_methods='avg'):
        ob = self.wbins.to(newbins.unit).value
        nb = newbins.value

        y = self.y.value
        ynew = rebin.rebin(nb, ob, y, 'avg') * self.y.unit

        if self.e is None:
            enew = None
        else:
            E = self.e.value * np.diff(ob)
            V = E ** 2
            Vnew = rebin.rebin(nb, ob, V, 'sum')
            enew = np.sqrt(Vnew) / np.diff(nb) * self.e.unit

        if self.other_data is None:
            other_data_new = None
        else:
            if type(other_data_bin_methods) is str:
                methods = [other_data_bin_methods] * len(self.other_data)
            other_data_new = {}
            for key, method in zip(self.other_data, methods):
                d = self.other_data[key]
                other_data_new[key] = rebin.rebin(nb, ob, d.value, method) * d.unit

        notes, refs = [copy.copy(a) for a in [self.notes, self.references]]
        newspec = Spectrum(None, ynew, err=enew, yname=self.ynames, notes=notes, references=refs, wbins=newbins,
                           other_data=other_data_new)
        return newspec

    def clip(self, wavemin, wavemax):
        keep = (self.wbins[1:] > wavemin) & (self.wbins[:-1] < wavemax)

        y = self.y[keep]

        wbins = self.wbins[:-1][keep]
        wmax = self.wbins[1:][keep][-1]
        wbins = np.append(wbins.value, wmax.value) * self.wbins.unit
        if wbins[0] < wavemin:
            wbins[0] = wavemin
        if wbins[-1] > wavemax:
            wbins[-1] = wavemax

        if self.e is None:
            e = None
        else:
            e = self.e[keep]

        notes = copy.copy(self.notes)
        return Spectrum(wbins, y, err=e, yname=self.ynames, notes=notes)

    def write(self, path, overwrite=False):
        if not path.endswith(Spectrum.file_suffix):
            path += Spectrum.file_suffix

        tbl = self.to_table()
        tbl.write(path, format=Spectrum.table_write_format, overwrite=overwrite)

    def step_line(self, return_err=False):
        """
        Returns w and y points that can be used to plot a stairstep version of the spectrum with plt.plot that is 
        better than using plt.step.
        
        Parameters
        ----------
        return_err : True | False
            If True, return an array for the error as a third variable.

        Returns
        -------
        wpts, ypts(, epts)
            Arrays of wavelength, y values, and (if return_err == True) error points to draw stairstep curves.
        
        """

        w, y = [np.empty(2 * len(self)) for _ in range(2)]
        w[::2], w[1::2] = self.wbins[:-1], self.wbins[1:]
        y[::2], y[1::2] = self.y
        if return_err:
            e = np.empty(2*len(self))
            e[::2],  e[1::2] = self.err
            return w, y, e
        else:
            return w, y

    @classmethod
    def read(cls, path_or_file_like):
        """
        Read in a spectrum.

        Parameters
        ----------
        path_or_file_like

        Returns
        -------
        Spectrum object
        """
        if type(path_or_file_like) is str and not path_or_file_like.endswith(cls.file_suffix):
            raise IOError('Can only read {} file.'.format(cls.file_suffix))

        tbl = table.Table.read(path_or_file_like, format='ascii.ecsv')
        w, dw, y = [tbl[s].quantity for s in ['w', 'dw', 'y']]
        tbl.remove_columns(['w', 'dw', 'y'])
        if 'err' in tbl.colnames:
            e = tbl['err']
            tbl.remove_column('e')
        else:
            e = None

        refs = tbl.meta['references']
        notes = tbl.meta['notes']
        ynames = tbl.meta['ynames']

        if len(tbl.colnames) > 0:
            other_data = {}
            for key in tbl.colnames:
                other_data[key] = tbl[key].quantity
        else:
            other_data = None

        spec = Spectrum(w, y, e, dw=dw, other_data=other_data, yname=ynames, references=refs, notes=notes)
        return spec

    @classmethod
    def blackbody(cls, T, wbins):
        # TODO make better by computing integral over bins
        # FIXME check the pi factor
        w = midpts(wbins)
        f = np.pi * 2 * const.h * const.c ** 2 / w ** 5 / (np.exp(const.h * const.c / const.k_B / T / w) - 1)
        f = f.to('erg s-1 cm-2 AA-1')
        return Spectrum(None, f, yname=['f', 'flux', 'surface flux'], wbins=wbins)


def midpts(a):
    return (a[:-1] + a[1:])/2.0


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