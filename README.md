
## Double-auto spectral baseline (das baseline)

Primary intended use is Raman spectroscopy. This is some modified version of Tikhonov regularization with two parameters: lambda (smoothing) and p (asymmetry). The starting point of the algorithm is described in the JRS paper: https://doi.org/10.1002/jrs.5952

The function ```find_both_optimal_parameters``` finds the optimal lambda and p for the given spectrum.
The function ```das_baseline``` is the main working function.

## A few notes on the algorithm
It uses wavelet transform to estimate the characteristic flexibility level of the baseline. With this, we can adjust the lambda. The asymmetry (p) is optimized based on the statistical analysis of the residuals.

## Structure:
```main.py``` -- Run it for the illustration. You can choose one of the three experimental sample spectra, given in the folder ```test_data```: graphene oxide Raman spectrum, hyper-Raman spectrum of water, Raman spectrum of achitin.

```expspec.py``` -- spectral class format, some supplementary function - probably you don't want to look at it.

```plot_the_funcamential.py``` -- plot the 2D functional, used to optimize the regularization parameters.


If you have your test spectrum as *numpy-readable* *x-y* file (a text file with two columns separated with tabs or spaces), you could use the following lines:
```python
from fit_multipeak import * # load the script
current_spectrum = np.genfromtxt('my_spectrum.txt') # read file to numpy format
testspec = ExpSpec(current_spectrum[:,0], current_spectrum[:,1]) # convert the spectrum to an *object* of a specific format.
_ = multipeak_fit_with_BL(testspec, saveresults = True) # fit it! The starting point will be generated automatically from *find_da_peaks* function.
```

The example above is a basic usage. You can set the starting point for the fit, specify the fitting range, control the display options etc.
There is a test spectrum and an example file for the starting point. You can try the following:
```python
from expspec import *  # load the class
import numpy as np  # necessary import
myspectrum = np.genfromtxt('path_to_my_spectrum/spectrumname.txt') # load your spectrum into numpy array
myspectrum = ExpSpec(myspectrum[:,0], myspectrum[:,1]) # convert it to the working format of the script
olam,op = find_both_optimal_parameters(testspec) # olam ~ optimal lambda, op ~ optimal p
```
With the above example you should get the complete (excessive) output with pictures in the folder ```current_output```.
