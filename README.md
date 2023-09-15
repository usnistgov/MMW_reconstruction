# SAR Reconstruction Code for MMW systems
- by Praful Gupta (UT Austin and Amazon)
- & Jack Glover (NIST)

We provide Python code for performing synthetic aperture radar (SAR)
reconstruction for millimeter wave (MMW) systems that use cylindrical geometry.

This code was created as part of the research described in [1] and is set up to reconstruct the
.ahi files distributed as part of the Department of Homeland Security's
Kaggle competition titled "Passenger Screening Algorithm Challenge" [2].
It can likely be modified to work for other systems that use a cylindrical
geometry.
The code was developed based on the information in [3] and [4].

The code allows the user to alter the reconstruction parameters, including
the range of frequencies that are used. 
This can allow the user to vary the quality of reconstructed images that are produced.
For example, the spatial resolution of the image can be controlled by adjusting the center frequncy (see below).

![image](https://github.com/usnistgov/MMW_reconstruction/assets/12698270/484e947e-3b2d-4bd3-92d5-45365d291f4c)


---

## References

[1] Gupta, P., Facktor, M. B., Glover, J. L., & Bovik, A. C. (2021, April). Validating the quality of millimeter-wave images input to deep-learning-based threat detection systems. In Automatic Target Recognition XXXI (Vol. 11729, pp. 221-229). SPIE. [LINK](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11729/117290Q/Validating-the-quality-of-millimeter-wave-images-input-to-deep/10.1117/12.2585639.full?SSO=1)

[2] https://www.kaggle.com/competitions/passenger-screening-algorithm-challenge

[3] Sheen, D., McMakin, D., & Hall, T. (2010). Near-field three-dimensional radar imaging techniques and applications. Applied Optics, 49(19), E83-E93.

[4] Soumekh, M. (1994). Fourier array imaging. Prentice-Hall, Inc..

