
py2shpss
========


.. image:: https://img.shields.io/pypi/v/py2shpss.svg
   :target: https://pypi.python.org/pypi/py2shpss
   :alt: 

.. image:: https://github.com/tachi-hi/py2shpss/workflows/CI/badge.svg
   :target: https://github.com/tachi-hi/py2shpss/actions?query=workflow%3ACI
   :alt: Build Status

.. image:: https://img.shields.io/pypi/l/py2shpss.svg
   :target: https://opensource.org/licenses/MIT
   :alt: 

A python implementation of two-stage HPSS (a singing voice extraction method)

Two-stage HPSS separates a music signal into harmonic, vocal, and percussive components.

License
-------

Copyright :copyright: 2020 Hideyuki Tachibana, `MIT License <https://github.com/tachi-hi/py2shpss/blob/master/LICENSE>`_

Usage
-----
Install
^^^^^^^

.. code-block:: bash

    pip install py2shpss

Code Example
^^^^^^^^^^^^

.. code-block:: python

	# import
	import numpy as np
	import scipy.io.wavfile as wavfile
	import py2shpss

	# load music
	sr, sig = wavfile.read("your_awesome_music.wav")
	if len(sig.shape) == 2:
		# mix left and right channels if stereo
		sig = np.mean(sig, axis=1)
	sig = sig / np.max(sig)

	# process and save
	twostageHPSS = py2shpss.twostageHPSS(samprate = sr)
	harmonic, vocal, percussive = twostageHPSS(sig)
	wavfile.write("vocal.wav", sr, vocal)



Citation
--------

Two-stage HPSS was proposed in following papers.

.. [1] \ H. Tachibana, T. Ono, N. Ono, S.Sagayama, **"Melody line estimation in homophonic music audio signals based on temporal-variability of melodic source,"** in *Proc. ICASSP*, pp.425--428, IEEE, 2010
.. [2] \ H. Tachibana, N. Ono, S. Sagayama, **"Singing voice enhancement in monaural music signals based on two-stage harmonic/percussive sound separation on multiple resolution spectrograms,"** *IEEE/ACM TASLP*, vol. 22, no. 1, pp. 228--237, 2014
.. [3] \ H. Tachibana, Y. Mizuno, N. Ono, S. Sagayama, **"A real-time audio-to-audio karaoke generation system for monaural recordings based on singing voice suppression and key conversion techniques"**, *Journal of Information Processing*, vol. 24, no. 3, pp. 470-482, Information Processing Society of Japan, 2016
