#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

directory_path = "data/raw/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000361/1.2.826.0.1.3680043.10.474.419639.426473907903631915363207391670/10-21-2002/"
topic = "covid"

x = 0
for i,j,y in os.walk(directory_path):
  os.mkdir("data/processed/" + topic + "/" + str(x));

  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(i)

  counter = 0
  for dicom_name in dicom_names:
    image = sitk.ReadImage(dicom_name)
    image = sitk.IntensityWindowing(image, -1000, 1000, 0, 255)
    image = sitk.Cast(image, sitk.sitkUInt8)
    sitk.WriteImage(image, "data/processed/" + topic + "/" + str(x) + "/" + str(counter) + ".png")
    counter += 1

  x = x + 1
