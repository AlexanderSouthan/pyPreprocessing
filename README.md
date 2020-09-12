# pyPreprocessing
For preprocessing of datasets, currently via baseline correction, smoothing,
filtering, transformation, and normalization. It relies on numpy, pandas,
scipy, tqdm and scikit-learn.

## Baseline correction (in baseline_correction.py)
Before baseline correction, data can be smoothed or tranformed by methods
described below. Implemented baseline correction methods are:
*Asymmetric Least Squares Smoothing (ALSS) described by P. Eilers and H.Boelens
(https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf)
*iALSS, based on *Anal. Methods*, **2014**, *6*, 4402–4407. The authors claim
that it is an improved ALSS method.
*Doubly reweighted penalized least squares (drPLS), based on *Applied Optics*,
**2019**, *58*, 3913-3920.
*SNIP, based on *Nuclear Instruments and Methods in Physics Research*,
**1988**, *934*, 396-402.
*ModPoly and IModPoly, algorithms based on polynomial fitting as described in
*Applied Spectroscopy*, **2007**, *61* (11), 1225-1232 and
Chemometrics and Intelligent Laboratory Systems, **2006**, *82*, 59– 65.
*Convex hull (based on https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data)

##Smoothing (in smooting.py)
Data can be extended on the edges by point mirroring to reduce smoothing
artifacts. Data output has the same dimensions like the input. In case of
unevenly spaced data, a possibility to interpolate the data to even spacing
is given that should be used. Currently implemented methods are:
*Savitzky-Golay
*Moving median
*Smoothing based on principal component analysis
*Selective moving average

##Filtering (in smoothing.py)
In contrast to smoothing, filtering mehods identify certain data points that
need to be dropped without any change of the remaining data. Current methods
are:
*Remove all values above a maximum threshold given.
*Remove all values below a minimum threshold given.
*A spike filter based on the selective moving average smoothing.

##Transformation (in transform.py)
Contains methods to apply or undo a transformation on the input data. 
Currently, the only implemented method is:
*LLS, which means log(log(sqrt(innput_data)))

##Normalization (in transform.py)
Normalize the input data. Currently only one method is implemented:
*Normalize the integral below the data to a certain value.

