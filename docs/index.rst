libra-toolbox
=============

``libra-toolbox`` is a library designed to facilitate the analysis of LIBRA experiments, including data processing, visualization, and model building.
This documentation will guide you through the various features and functionalities of libra-toolbox.

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: Neutron spectrum analysis
        :img-bottom: _static/neutron_spectrum.png
        :link: examples/diamond_detector.ipynb
        :text-align: center

    .. grid-item-card:: Tritium release model
        :img-bottom: _static/tritium_model.png
        :link: examples/tritium_model.ipynb
        :text-align: center

    .. grid-item-card:: Parametric optimisation
        :img-bottom: _static/optimisation.png
        :link: examples/fit_tritium_release.ipynb
        :text-align: center

    .. grid-item-card:: PRT analysis
        :img-bottom: _static/prt.png
        :link: examples/prt.ipynb
        :text-align: center

Installation
------------

To install libra-toolbox, use pip:

.. code-block:: bash

   pip install libra-toolbox

For more details, see the :ref:`installation` page.

Roadmap
-------
.. role:: strike
    :class: strike

- :strike:`Tritium model: 0D model for tritium release`
- :strike:`Neutron detection: tools and scripts for neutron detection post processing`
- :strike:`Neutronics: tools to facilitate building OpenMC models`
- Tritium transport: tools to facilitate building `FESTIM <https://github.com/festim-dev/festim>`_ tritium transport models
- :strike:`Code coverage with Codecov`
- Conda package


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples/index
   modules/index
