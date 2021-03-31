#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

def command_iteration(method):
  print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

def segmentation(fixed_file_name, moving_file_name, output_file_name):
  fixed = sitk.ReadImage(fixed_file_name, sitk.sitkFloat32)
  moving = sitk.ReadImage(moving_file_name, sitk.sitkFloat32)

  R = sitk.ImageRegistrationMethod()
  R.SetMetricAsMeanSquares()
  R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
  R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
  R.SetInterpolator(sitk.sitkLinear)

  R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

  outTx = R.Execute(fixed, moving)

  print("-------")
  print(outTx)
  print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
  print(f" Iteration: {R.GetOptimizerIteration()}")
  print(f" Metric value: {R.GetMetricValue()}")

  sitk.WriteTransform(outTx, output_file_name)

  if ("SITK_NOSHOW" not in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    sitk.Show(cimg, "ImageRegistration1 Composition")

segmentation("data/processed/54.png", "data/processed/54.png", "data/results/output.txt")