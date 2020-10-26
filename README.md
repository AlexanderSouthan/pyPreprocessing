# pyPreprocessing
For preprocessing of datasets like Raman spectra, infrared spectra, UV/Vis
spectra, but also HPLC data and many other types of data, currently via
baseline correction, smoothing, filtering, transformation, normalization and
derivative. It relies on numpy, pandas, scipy, tqdm and scikit-learn, but also
on https://github.com/AlexanderSouthan/pyRegression for the introduction of
equality constraints into the polynomial baseline estimation methods.

## Baseline correction (in baseline_correction.py)
Before baseline correction, data can be smoothed or transformed by methods
described below. Implemented baseline correction methods are:
* Asymmetric Least Squares Smoothing (ALSS) described by P. Eilers and H.Boelens
(https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf).
Requires two paramters that have to be tuned to optimize the result. It can
take a while to optimize the parameters, but then the results are reasonable
most of the time. 
* iALSS, based on *Anal. Methods*, **2014**, *6*, 4402–4407. The authors claim
that it is an improved ALSS method. However, it has one more paramter compared
to ALSS that has to be optimized manually.
* Doubly reweighted penalized least squares (drPLS), based on *Applied Optics*,
**2019**, *58*, 3913-3920. Has two parameters controlling the result that have
to be tuned manually. Deals rather nice with noisy spectra if the parameters
were chosen well.
* SNIP, based on *Nuclear Instruments and Methods in Physics Research*,
**1988**, *934*, 396-402. Does not require any guess on a parameter, so seems
quite interesting. However has problems with convex baselines and is not very
robust in general.
* ModPoly and IModPoly, algorithms based on polynomial fitting as described in
*Applied Spectroscopy*, **2007**, *61* (11), 1225-1232 and
*Chemometrics and Intelligent Laboratory Systems*, **2006**, *82*, 59– 65. They
are both quite robust polynomial fitting algorithms that require only input on
the polynomial order. IModPoly deals better with noise, but this can also be
handled with ModPoly if the spectra are smoothed before baseline calculation.
Both allow equality constraints in the fit so that the baseline passes specific
points or has certain slopes at specific wavenumbers.
* Convex hull (based on https://dsp.stackexchange.com/questions/2725/
how-to-perform-a-rubberband-correction-on-spectroscopic-data). Quite
interesting method that does not require any initial guess, but requires a
convex baseline.
* Piecewise polynomial fitting (PPF). This method relies on the IModPoly
algorithm and segments the spectrum into baseline separated areas (given by
user input) which are then fitted one by one. Discontinuities between the
polynomial baseline segments can be reduced by applying equality constraints
on the fits.

## Smoothing (in smooting.py)
Data can be extended on the edges by point mirroring to reduce smoothing
artifacts. Data output has the same dimensions like the input. In case of
unevenly spaced data, a possibility to interpolate the data to even spacing
is given that should be used. Currently implemented methods are:
* Savitzky-Golay
* Moving median
* Smoothing based on principal component analysis
* Weighted moving average

## Filtering (in smoothing.py)
In contrast to smoothing, filtering mehods identify certain data points that
need to be dropped without any change of the remaining data. Current methods
are:
* Remove all values above a maximum threshold given.
* Remove all values below a minimum threshold given.
* A spike filter based on the selective moving average smoothing.

## Transformation (in transform.py)
Contains methods to apply or undo a transformation on the input data. 
Currently, the only implemented method is:
* LLS, which means log(log(sqrt(input_data)))

## Normalization (in transform.py)
Normalize the input data. Currently only one method is implemented:
* Normalize the integral below the data to a certain value.

## Derivative (in num_derive.py)
Calculate the numerical derivative of data. Different derivative orders can be
obtained.