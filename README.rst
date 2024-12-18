managementMetadataFromRS
========================

This project contains application source code for manage and add metadata to remote sensing images. Basically, the package has a main module (core) and submodules (ReSSIM and CenterPivot) to  serching specific targets how burned areas or center pivot irrigation areas in remote sensing images.

Release Notes
-------------

- Support CBERS 4 (MUX) and LANDSAT 7 (ETM+)/8 (OLI) files with single bands

Installation
------------

**Dependencies**

    Python 3.7.X and Numpy,OpenCV/GDAL
    

Build Steps
-----------

**Setup Conda Environment** 

With Conda installed [#]_, run::

  $ git clone  https://gitlab.dpi.inpe.br/managementMetadataFromRS/managementMetadataFromRS.git
  $ cd managementMetadataFromRS
  $ make install
  $ source activate managementMetadataFromRS

.. [#] If you are using a git server inside a private network and are using a self-signed certificate or a certificate over an IP address ; you may also simply use the git global config to disable the ssl checks::

  git config --global http.sslverify "false"


Usage
-----

See core.py --help for command line details.


Data Processing Requirements
----------------------------

This version of the application requires the input files to be in the GeoTIFF format, compressed or not with zip or gzip.


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the National Institute for Space Research (INPE). No warranty, expressed or implied, is made by the INPE or the Brazil Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the INPE nor the Brazil Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.


Licence
-------

MIT License

Copyright (c) 2019 Marcos Rodrigues

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Authors
-------

`managementMetadataFromRS` was written by `Marcos Rodrigues <marcos.rodrigues@inpe.br>`_.
