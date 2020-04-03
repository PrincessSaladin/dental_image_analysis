# coding:utf-8
import math
import numpy as np
import pandas as pd
from collections import namedtuple

"""
The struct containing the error summary.
"""
ErrorSummary = namedtuple('ErrorSummary',
                          'mean_error median_error max_error min_error point_distance_list l2_norm_error_list sorted_index_list')
"""
The struct containing the benchmark summary.
"""
BenchmarkSummary = namedtuple('BenchmarkSummary',
                              'mean_diff median_diff max_diff min_diff l2_norm_diff_list sorted_index_list')
"""
Run error analysis and return the error statistics summary.
Input arguments:
  labelled_point_dict: A dict whose keys and values are filenames and coordinates of labelled points respectively.
  detected_point_dict: A dict whose keys and values are filenames and coordinates of detected points respectively.
  descending:          Flag indicating whether errors sorted in ascending or descending order.
Return:
  error_summary:       Summary of error statistics.
"""
def ErrorAnalysis(labelled_point_dict, detected_point_dict, descending=True):
  truly_positive_files = list(set(labelled_point_dict.keys()).intersection(set(detected_point_dict.keys())))
  truly_positive_files.sort()

  labelled_point_list = []
  detected_point_list = []
  for file_name in truly_positive_files:
    labelled_point_list.append(labelled_point_dict[file_name])
    detected_point_list.append(detected_point_dict[file_name])

  labelled_point_list = np.array(labelled_point_list)
  detected_point_list = np.array(detected_point_list)

  point_distance_list = detected_point_list - labelled_point_list
  l2_norm_error_list = np.linalg.norm(point_distance_list, axis = 1)
  max_error = np.amax(l2_norm_error_list)
  min_error = np.amin(l2_norm_error_list)
  median_error = np.median(l2_norm_error_list)
  mean_error = np.mean(l2_norm_error_list)

  # By default, argsort() sorts an array in ascending order.
  sorted_index_list = np.argsort(l2_norm_error_list)
  if descending:
    sorted_index_list = sorted_index_list[::-1]

  error_summary = ErrorSummary(
    mean_error = mean_error,
    median_error = median_error,
    max_error = max_error,
    min_error = min_error,
    point_distance_list = point_distance_list,
    l2_norm_error_list = l2_norm_error_list,
    sorted_index_list = sorted_index_list)
  return error_summary, truly_positive_files


"""
Run error analysis and return the error statistics summary.
Input arguments:
  labelled_point_dict:          A dict whose keys and values are filenames and coordinates of labelled points respectively.
  baseline_detected_point_dict: A dict whose keys and values are filenames and coordinates of 
                                baseline detected points respectively.
  experiment_detected_point_dict: A dict whose keys and values are filenames and coordinates of experiment 
                                detected points respectively.
  descending:                   Flag indicating whether errors sorted in ascending or descending order.
Return:
  error_summary:                Summary of benchmark statistics.
"""
def BaselineBenchmark(labelled_point_dict, baseline_detected_point_dict, experiment_detected_point_dict, descending=True):
  intersection_set = set(baseline_detected_point_dict.keys()).intersection(set(experiment_detected_point_dict.keys()))
  truly_positive_files = list(set(labelled_point_dict.keys()).intersection(intersection_set))
  truly_positive_files.sort()
  labelled_point_list = []
  baseline_point_list = []
  experiment_point_list = []

  for file_name in truly_positive_files:
    labelled_point_list.append(labelled_point_dict[file_name])
    baseline_point_list.append(baseline_detected_point_dict[file_name])
    experiment_point_list.append(experiment_detected_point_dict[file_name])

  labelled_point_list = np.array(labelled_point_list)
  baseline_point_list = np.array(baseline_point_list)
  experiment_point_list = np.array(experiment_point_list)

  point_distance_list1 = baseline_point_list - labelled_point_list
  l2_norm_error_list1 = np.linalg.norm(point_distance_list1, axis=1)

  point_distance_list2 = experiment_point_list - labelled_point_list
  l2_norm_error_list2 = np.linalg.norm(point_distance_list2, axis=1)

  max_error1 = np.amax(l2_norm_error_list1)
  min_error1 = np.amin(l2_norm_error_list1)
  median_error1 = np.median(l2_norm_error_list1)
  mean_error1 = np.mean(l2_norm_error_list1)
  max_error2 = np.amax(l2_norm_error_list2)
  min_error2 = np.amin(l2_norm_error_list2)
  median_error2 = np.median(l2_norm_error_list2)
  mean_error2 = np.mean(l2_norm_error_list2)

  l2_norm_diff_list = l2_norm_error_list2 - l2_norm_error_list1

  # Sort the volumes based on descending error downgrade.
  # By default, argsort() sorts an array in ascending order.
  sorted_index_list = np.argsort(l2_norm_diff_list)
  if descending:
    sorted_index_list = sorted_index_list[::-1]

  error_summarys = []

  error_summary1 = ErrorSummary(
    mean_error=mean_error1,
    median_error=median_error1,
    max_error=max_error1,
    min_error=min_error1,
    point_distance_list=point_distance_list1,
    l2_norm_error_list=l2_norm_error_list1,
    sorted_index_list=sorted_index_list)
  error_summarys.append(error_summary1)

  error_summary2 = ErrorSummary(
    mean_error=mean_error2,
    median_error=median_error2,
    max_error=max_error2,
    min_error=min_error2,
    point_distance_list=point_distance_list2,
    l2_norm_error_list=l2_norm_error_list2,
    sorted_index_list=sorted_index_list)
  error_summarys.append(error_summary2)

  benchmark_summary = BenchmarkSummary(
    mean_diff = mean_error2 - mean_error1,
    median_diff = median_error2 - median_error1,
    max_diff = max_error2 - max_error1,
    min_diff = min_error2 - min_error1,
    l2_norm_diff_list=l2_norm_diff_list,
    sorted_index_list=sorted_index_list)
  return error_summarys, benchmark_summary, truly_positive_files