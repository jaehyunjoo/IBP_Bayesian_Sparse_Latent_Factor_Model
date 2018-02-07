# Bayesian Sparse Latent Factor Model (work in pregress)

**Currently, this code runs using the fixed number of latent factors, utilizing an IBP prior only for imposing sparseness on factor loadings.**

Bayesian Sparse Latent Factor Model using an IBP prior with a nonparametric modeling of the latent factor distribution.

This code tries to implement a sparse multivriate latent factor model, with an extension for regression components based on covariates. It follows the model specifications in [Lucas et al.(2006)](http://ftp.stat.duke.edu/WorkingPapers/06-01.pdf) and [Carvalho et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3017385/) but I attempted to use an IBP prior for a posterior inference on the number of latent factors instead of using a spike-slab prior used in the papers. 

This model imposes explicit sparseness on the factor loading matrix. A simple linear-gaussian model imposing sparseness on latent factors (factor scores) rather than on factor loadings is available in [IBP_Linear_Gaussian_Latent_Factor_Model](https://github.com/jaehyunjoo/IBP_Linear_Gaussian_Latent_Factor_Model).

The IBP-based factor model components were constructed based on [Knowles and Ghahrmani (2011)](https://www.jstor.org/stable/23024862?seq=1#page_scan_tab_contents) and its [MATLAB code](https://github.com/davidaknowles/nsfa).

The nonparametric extension on latent factors (factor scores) for adaptating non-Gaussianity in data was coded following the specifications in [Carvalho et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3017385/) (utilize a Dirichlet Process Mixture Model on latent factors) but I was not sure I did the math correctly and this python code is a little slow. It needs to be updated and validated.

For example usage, run

```python
python demo.py
```
This demo is based on a simulated data set consisting of 6x6 images in Griffiths and Ghahramani (2011). The default setting assumes Gaussian latent factors. 

The demo output will be saved as figures using [David Andrzejewski's code](https://github.com/davidandrzej/PyIBP) (scaledimage.py) that mimics MATLAB imagesc().

