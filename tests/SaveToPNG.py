#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

# the directory should contain .dcm files
directory_path = "data/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/1.2.826.0.1.3680043.10.474.419639.312580455409613733097488204614/08-02-2002/1.2.826.0.1.3680043.10.474.419639.108518937868403894887894311320"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(directory_path)

image = sitk.ReadImage(dicom_names[50])
image = sitk.IntensityWindowing(image, -1000, 1000, 0, 255)
image = sitk.Cast(image, sitk.sitkUInt8)
sitk.WriteImage(image, "output.png")