.. GAN documentation master file, created by
   sphinx-quickstart on Fri Jun 30 10:53:18 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GAN's documentation!
===============================

.. toctree::
   :maxdepth: 2

   tutorial

Overview of GANs
================
Generative Adversarial Networks (GANs) are neural networks which use two components, a *generator* and a *discriminator*, to create a mapping from a latent space to a data space.  

The generator (G) is responsible for creating the data, and the discriminator is responsible for determining a given input is created by the generator or is actual training data.  By using this approach, it is possible for the generator portion of the GAN to create the mapping from the latent space to the data space in an unsupervised fashion.  However, during training, the discriminator is provided a signal which indicates whether or not the given input is from G or from the training data.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
