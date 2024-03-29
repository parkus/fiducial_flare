`fiducial_flare` is a package for generating a reasonable approximation of the UV emission of M dwarf stars over a single flare or a series of them. The simulated radiation is resolved in both wavelength and time. The intent is to provide consistent input for applications requiring time-dependent stellar UV radiation fields that balances simplicity with realism, namely for simulations of exoplanet atmospheres.

For this balance of simplicity and realism, the flares generated are idealized in the spectral and temporal distribution of their energy through the following assumptions:

- The energy budget of the flares is constant. It was compiled by Loyd et al. 2018 and is given in "relative_energy_budget.ecsv".
- The NUV continuum is taken to be a 9,000 K blackbody with energy scaled against the Si IV doublet per the energy budgets of Hawley et al. 2003.
- Some strong but unobserved lines are assumed to show the same response (relative to quiescent levels) as a proxy line with good observations and a similar formation temperature. Specified with the notation "unobserved line -> proxy," these are
  + Lya core -> O I 1305
  + Lyb -> O I 1305
  + Lyg -> O I 1305
  + Mg II 2796, 2804 -> O I 1305
  + Al II 1670 -> C II 1334,1335
  + O VI 1031,1037 -> N V 1238,1242
- The temporal evolution of flux is taken to a be a boxcar followed by exponential decay, following the formula given in Loyd et al. 2018.
- Flare energies are distributed as a power-law based on the fit to M dwarf Si IV 1394,1403 flares of Loyd et al. 2018.
- The flare rate is constant.
- Flare events follow a Poisson distribution (implying an exponential distribution in flare waiting times, e.g. Wheatland 2000)

To maintain comparability between works that use input generated by this package, we will attempt to minimize future changes and document any we do make well.

### Tips
#### Units
`fiducial_flare` relies on the `astropy.units` module to ensure unit consistency, which some users might not be familiar with. In `fiducial_flare`, all dimensional quantities must be `astropy.units.Quantity` objects, i.e. numbers with units. These can be formed by simply multiplying a number by a unit from the `astropy.units` module. For example, to specify a time of 100 s, simply do

```python
from astropy import units as u
t = 100 * u.s
```

Sometimes `Quantity` objects don't play nicely with `numpy`. This tends to result in some awkward usage of `x.value` (to get the float value with no unit), where `x` is a `Quantity`.

For more information on units with `astropy`, see [the docs](http://docs.astropy.org/en/stable/units/) for the `units` module.

#### Integration
Be aware that `fiducial_flare` integrates over spectral and temporal bins to ensure that energy is conserved when bins are changed. In contrast, it is standard practice for many to compute spectral density or rates at the midpoint of a wavelength or time bin. `fiducial flare` integrates the value over the bin and divides by the bin width to give the average value.

### Referencing This Work (Credit)
Please acknowledge your use of this work by citing Loyd et al. 2018  (https://ui.adsabs.harvard.edu/abs/2018ApJ...867...71L) and specifying the version (`flare_simulator.version`).

### Quick(ish) Start
Let's generate some input you could use to drive a photochemical model of a planetary atmosphere, starting with some imports.

```python
import numpy as np
from astropy import units as u
from astropy import table
from matplotlib import pyplot as plt
plt.ion()
from fiducial_flare import flare_simulator as flaresim
```

As the base quiescent SED, we will use the MSUCLES spectrum of GJ 832. You can get the FITS version of that spectrum [here](https://archive.stsci.edu/missions/hlsp/muscles/gj832/hlsp_muscles_multi_multi_gj832_broadband_v22_adapt-const-res-sed.fits) and read more about this SED with caveats and such [here](https://archive.stsci.edu/missions/hlsp/muscles).

```python
# load in the quiescent SED
path_sed = 'Downloads/hlsp_muscles_multi_multi_gj832_broadband_v22_adapt-const-res-sed.fits'
sed = table.Table.read(path_sed)
```

This SED has way higher resolution that we need. In fiducial flare spectrum, lines are represented by single 200 km/s wide bins (about 1 Å in the FUV). Higher resolutions will yield odd-looking output once we add flare spectra to the SED. We'll rebin to 1 Å bins from 100-3000 Å and then 10 Å through the 5.5 µm limit of the SED. If you are using simulation software that is hardcoded to a specific wavelength grid, you will want to use that grid instead. 

The original SED also has variable binning. The left and and right edges of the bins are given in the 'WAVELENGTH0' and 'WAVELENGTH1' columns. 

The rebin function is one of the few in fiducial_flare that specifically requires input *without* units.

```python
wbins_sed = np.append(sed['WAVELENGTH0'], sed['WAVELENGTH1'][-1]) * u.AA

wbins_fuv = np.arange(100, 3000, 1) * u.AA
wbins_red = np.arange(3000, 5.5e4, 10) * u.AA
wbins = np.hstack((wbins_fuv, wbins_red))

# now rebin
Fq_bolo = flaresim.rebin(wbins.value, wbins_sed.value, sed['BOLOFLUX'])

# add the proper units since rebin can't work with them
Fq_bolo = Fq_bolo * u.Unit('AA-1')
```

The flux here is normalized by the bolometric luminosity of the star (units of Å-1). For this example, let's consider a planet receiving the same instellation as Earth (~1300 W m-2). To get that value, we just need to multiply by the desired instellation.

```python
F_bolo_earth = 1300 * u.Unit('W m-2')
Fq = Fq_bolo * F_bolo_earth
Fq = Fq.to('erg s-1 cm-2 AA-1')
```

Now that we've got the quiescent SED all prepped, let's simulate some flares to go on top of that. The default flare parameters, provided in the `flaressim.flare_defaults` dictionary, are appropriate for simulating UV flares occuring on an early-mid M star. Blackbody emission is added to simulate flux emitted at longer wavelengths. 

Probably the most important parameter is the quisecent flux of the star in the Si IV 1393,1402 Å emission line doublet, since the flares are scaled to this value. The default value is appropriate for a field-age M star at the distance of a habitable-zone planet. Alternatively, we can just measure the actual value by rebinning (integrating) our quiescent SED over the range of those lines. You will want to be sure your SED has an accurate Si IV flux if you go this route!

```python
wbin_SiIV = [1390, 1410] * u.AA
Fq_SiIV = flaresim.rebin(wbin_SiIV.value, wbins.value, Fq.value) * u.Unit('erg s-1 cm-2 AA-1')

# this actuyally spits out the flux density, but what we want is the flux
Fq_SiIV = Fq_SiIV * np.diff(wbin_SiIV)
```

We need some time bins to simulate flares over. We will use 60 s bins covering a full day.

```python
# sadly arange can't handle unit input
tbins = np.arange(0, 24*60*60, 60) * u.s
```

After all that set up, simulating an actual time series of fluxes from a random series of flares is as simple as. For reproducability, we will set a `np.random.seed` value.

```python
np.random.seed(42)
Ff = flaresim.flare_series_spectra(wbins, tbins, SiIV_quiescent=Fq_SiIV)
```

The resulting array has dimensions of (no. time bins) x (no. wavelength bins), in this case 1439x8099. Each row (e.g. `Ff[0,:]`) is a spectrum for the corresponding time bin. 

To get the spectrum the planet will actually see, we need to add these spectra onto the quiescent SED.

```python
Ftot = Fq[None,:] + Ff
```

Now let's have a look at what we've accomplished by plotting spectra at the highest peak, at quiescence and a smattering of other times (which will mostly fall near quiescence).

```python
plt.figure()

# quiescence
lnq, = plt.step(wbins[:-1], Fq, where='pre', label='quiescence')

# highest peak
imax = np.argmax(Ff[:,0])
lnp, = plt.step(wbins[:-1], Ftot[imax,:], where='pre', label='max')

# random times
n = 20
step = len(Ftot) // n
for F in Ftot[::step, :]:
  ln, = plt.step(wbins[:-1], F, where='pre', color='k', lw=0.5, alpha=0.5, label='smattering')

plt.legend(handles=(lnq, lnp, ln))
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux (erg s-1 cm-2 Å-1)')
plt.yscale('log')
```

Note that the flares merge with the quiescent spectrum toward the optical. This is as it should be. Their relative effect is much stronger in the UV and especially FUV than in the optical. Flares causing a percent change in the optical can correspond to changes of several orders of magnitude in FUV emission. We can plot the time series from one of the FUV bins as well as an optical bin to visualize the flares that the code randomly generated and see how their effects differ between wavelengths. You will have to zoom in to see the flares at optical wavelengths. 

```python
plt.figure()

i1393 = np.argmin(np.abs(wbins - 1393*u.AA)) # one of the components of the Si IV doublet
i5000 = np.argmin(np.abs(wbins - 5000*u.AA))

plt.step(tbins[:-1], Ftot[:,i1393]/Fq[i1393], where='pre', label='Si IV (1393 Å)')
plt.step(tbins[:-1], Ftot[:,i5000]/Fq[i5000], where='pre', label='Optical (5000 Å)')
plt.xlabel('Time (s)')
plt.ylabel('Relative Flux')
plt.yscale('log')
plt.legend()
```

You could now feasibly supply the Ftot array to atmospheric simulation code in order to simulate the effects of flare radiation. 



#### More Examples
```python
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
plt.ion()
from fiducial_flare import flare_simulator as flaresim

# --------
# generate spectra of a flare through various phases

# Let's look at a flare with an equivalent duration of 1 Ms occuring 100 s after the start of observations
eqd = 1*u.Ms
t0 = 100.*u.s

# first, a look at a well-resolved lightcurve
tbins_lc = np.linspace(0, 3000., 1000)*u.s
y = flaresim.flare_lightcurve(tbins_lc, t0, eqd)
plt.figure()
plt.step(tbins_lc[:-1], y, where='post')
plt.xlabel('Time [s]')
plt.ylabel('Quiescent-Normalized Flux')
plt.title('Lightcurve')

# let's get a spectrum from the impulsive phase and a few integrating
# intervals of the decay phase
tbins_spectra = [100., 660., 900., 1300., 2500.]*u.s
names = ['Impuslive', 'Decay 1', 'Decay 2', 'Decay 3']

# show these on the lightcurve plot
[plt.axvline(t.value, color='r', alpha=0.5) for t in tbins_spectra]
tmids = (tbins_spectra[:-1] + tbins_spectra[1:])/2
for tmid, name in zip(tmids, names):
    plt.annotate(name, xy=(tmid.value, 0.5), xycoords=('data', 'axes fraction'),
                 ha='center', va='center', rotation='vertical')

# now let's get some spectra
wbins = np.arange(912, 5000., 1.) * u.AA
spectra = flaresim.flare_spectra(wbins, tbins_spectra, t0, eqd)

# and plot them
plt.figure()
for spectrum in spectra:
    plt.step(wbins[:-1], spectrum, where='post')
plt.yscale('log')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Flux Density [erg s$^{-1}$ cm$^{-1}$ AA$^{-1}$]')
plt.legend(names)

# Aside: An important parameter is the quiescent Si IV flux, as this determines
# the absolute fluxes of the flare. A default value (based on an Earth-equivalent
# instellation from GJ 832) is supplied. That value is
print(flaresim.flare_defaults['SiIV_quiescent'])
# but for now we've just used the default.


# --------
# get a lightcurve  of  spectral  energy density within
# a single bandpass for a  flare

# let's go with a 10 ks flare starting 100 s into the
# "observation"
t0 = 100. * u.s
eqd = 10 * u.ks

# we need to set the absolute flux level  of the flare
# this is done by setting  the  Si IV flux, since
# everything is  normalized to Si IV. Let's go with
# 1 erg/s/cm2 (reasonable for a planet orbiting an
# active star with T_eq = 270 K)
flare_params = flaresim.flare_defaults.copy()
flare_params['SiIV_quiescent'] = 1*u.Unit('erg s-1 cm-2')

# make some time bins. Let's go coarse
tbins = np.arange(0, 1000, 60.) * u.s

# pick  the bandpass. How about all of the FUV?
wbins = [912, 1700] * u.AA

# get time-evolving spectra
spectra = flaresim.flare_spectra(wbins, tbins, t0, eqd)

# but there is only one broad wavelength, so get rid
# of the wavelength  dimension
flux_density = np.squeeze(spectra)

# plot it up!
plt.figure()
plt.step(tbins[:-1], flux_density, where='post')
plt.xlabel('Time [s]')
plt.ylabel('{} Flux Density (erg s-1 cm-2 AA-1)'.format(wbins))
# --------



# --------
# create and plot the standard lightcurve for a flare
# with a 10 ks equivalent duration.
tbins = np.linspace(0, 1000, 1000) * u.s
t0 = 100. * u.s
eqd = 10 * u.ks
y = flaresim.flare_lightcurve(tbins, t0, eqd)

# plot
plt.figure()
plt.step(tbins[:-1], y, where='post')
plt.xlabel('Time [s]')
plt.ylabel('Quiescent-Normalized Flux')

# verify that area under curve equals eqd
dt = np.diff(tbins)
area = np.sum(dt*y)
print(area)
# --------



# --------
# plot the standard energy budget spectral density for the fiducial flare

# note that the spectrum is constructed relative to the *integrated value*
# ofSi IV, so whether a flux density, energy spectral density, etc. is
# returned depends on what is supplied for the Si IV value

wbins = np.arange(912, 5000., 1.) * u.AA
si4_fluence = 1.0
e = flaresim.flare_spectrum(wbins, si4_fluence)

plt.figure()
plt.step(wbins[:-1], e, where='post')
plt.yscale('log')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Normalized Spectral Energy Density [$\AA^{-1}$]')
# --------



# --------
# generate a random series of flares from the default FFD and plot the lightcurve for them
time_span = 1e5*u.s
tbins = np.arange(0, time_span.value, 10.) # astropy quantities don't work here
tbins = tbins*u.s
y, flares = flaresim.flare_series_lightcurve(tbins, return_flares=True)

# first, let's have a look at what flares we drew
t_flare, eqds = flares
from astropy import table
flare_tbl = table.Table((t_flare, eqds), names=('Start Time', 'Equivalent Duration'))
print(flare_tbl)

# now what does the lightcurve of these events look like?
plt.figure()
plt.step(tbins[:-1], y, where='post')
plt.xlabel('Time [s]')
plt.ylabel('Quiescent-Normalized Flux')
# --------



# --------
# modify the standard flare to have a blackbody of 10,000 K instead of a 9,000 K
# (but the same ratio of blackbody to Si IV energy)

# let's start with the default flare parameters
flare_params = flaresim.flare_defaults.copy()

# what can we play with?
print(flare_params.keys())

# change the BB temperature to 10,000 K
flare_params['T_BB'] = 1e4 * u.K

# we should also turn off the default clipping of the blackbody
# in the FUV and shortward since it will contribute significantly
# more flux there than is presumed to be included in the fiducial
# flare energy budget
flare_params['clip_BB'] = False

# let's see what the revised spectral energy budget looks like
wbins = np.arange(912, 5000., 1.) * u.AA
si4_fluence = 1.0

## we could supply our custom flare_params dictionary
e = flaresim.flare_spectrum(wbins, si4_fluence, **flare_params)
## - or - we could specify a blackbody temperature and turn off
## clipping as keywords
e = flaresim.flare_spectrum(wbins, si4_fluence, T_BB=1e4*u.K, clip_BB=False)

plt.figure()
plt.step(wbins[:-1], e, where='post')
plt.yscale('log')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Normalized Spectral Energy Density [$\AA^{-1}$]')
# --------

```
