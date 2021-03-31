#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

# the directory should contain .dcm files
directory_path = "data/raw/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/1.2.826.0.1.3680043.10.474.419639.312580455409613733097488204614/08-02-2002/1.2.826.0.1.3680043.10.474.419639.108518937868403894887894311320"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print("Image size:", size[0], size[1], size[2])
sitk.Show(image, "Dicom Series")