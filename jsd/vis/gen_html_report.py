# coding:utf-8
import numpy as np
import os
import webbrowser
from jsd.vis.error_analysis import ErrorAnalysis
from jsd.vis.error_analysis import BaselineBenchmark


def AddDocumentText(original_text, new_text_to_add):
  return original_text + r'+"{0}"'.format(new_text_to_add)

"""
Write the texts to a html file.
"""
def WriteToHtmlReportFile(document_text, analysis_text, html_report_path, width):
  f = open(html_report_path, 'w')
  message = """
    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <title>result analysis</title>
      <style type="text/css">
          *{
              padding:0;
              margin:0;
          }
          .content {
              width: %spx;
              z-index:2;
          }
          .content img {
              width: %spx;
              transition-duration:0.2s;
              z-index:1;
          }
          .content img:active {
              transform: scale(2);
              -webkit-transform: scale(2); /*Safari 和 Chrome*/
              -moz-transform: scale(2); /*Firefox*/
              -ms-transform: scale(2); /*IE9*/
              -o-transform: scale(2); /*Opera*/
          }
      </style>
    </head>
    <body>
      <script type="text/javascript">       
        document.write(%s)               
      </script>
      <h1> Summary:</h1>
      %s
    </body>
    </html>""" % (width, width, document_text, analysis_text)

  f.write(message)
  f.close()
  # webbrowser.open(html_report_path, new = 1)


"""
Generate landmark evaluation HTML report.
Input arguments:
  file_name_list:   The list of 3D volume file names including postfix.
  point_dicts:      A list of dicts containing the labelled(detected,experiment) points' coordinates.
  additional_info:  A dict containing the information of no_landmark, false positives 
                    and miss detections.
  usage_flag:       A integer indicating the usage of the html tool.
                    1 for ground truth only
                    2 for error analysis
                    3 for benchmark comparison
  output_folder:    The output folder containing the html report.
  image_folder:     The image folder containing the captured 2D plane images.
                    Must be a relative path to the output_folder.
  html_report_name: The HTML report file name.
"""
def GenHtmlReport(file_name_list, point_dicts, additional_info, usage_flag,
                  output_folder, image_folder='./images',
                  html_report_name='result_analysis.html'):
  # how many cases are involved.
  num_data = len(file_name_list)

  labelled_point_dict = point_dicts[0]
  width = 500
  # find the unlabelled cases.
  tmp_set = set(file_name_list) - set(labelled_point_dict.keys())
  additional_info.no_lm.extend(list(tmp_set))
  additional_info.no_lm.sort()
  image_link_template = r"<div class='content'><img border=0  src= '{0}'  hspace=1  width={1} class='pic'></div>"
  error_info_template = r'<b>Labelled</b>: [{0:.2f}, {1:.2f}, {2:.2f}];  '
  # truly_positive_files is defined to contain the normal cases(which have nothing to do
  # with unlabelling, false positives and miss detections.)
  truly_positive_files = list(labelled_point_dict.keys())

  # if used for error analysis.
  if usage_flag == 2:
    detected_point_dict = point_dicts[1]
    width = 300
    tmp_set = set(labelled_point_dict.keys()) - set(detected_point_dict.keys())
    additional_info.fn1.extend(list(tmp_set))
    additional_info.fn1.sort()
    tmp_set = set(detected_point_dict.keys()) - set(labelled_point_dict.keys())
    additional_info.fp1.extend(list(tmp_set))
    additional_info.fp1.sort()
    error_info_template += r'<b>Detected</b>: [{3:.2f}, {4:.2f}, {5:.2f}];  '
    error_info_template += r'<b>Error</b>: x:{6:.2f}; y:{7:.2f}; z:{8:.2f}; L2:{9:.2f}'
    error_summary, truly_positive_files = ErrorAnalysis(labelled_point_dict, detected_point_dict)

  # if used for benchmark comparison with baseline model.
  if usage_flag == 3:
    baseline_point_list = point_dicts[1]
    experiment_point_list = point_dicts[2]
    width = 200
    tmp_set = set(labelled_point_dict.keys()) - set(baseline_point_list.keys())
    additional_info.fn1.extend(list(tmp_set))
    additional_info.fn1.sort()
    tmp_set = set(baseline_point_list.keys()) - set(labelled_point_dict.keys())
    additional_info.fp1.extend(list(tmp_set))
    additional_info.fp1.sort()
    tmp_set = set(labelled_point_dict.keys()) - set(experiment_point_list.keys())
    additional_info.fn2.extend(list(tmp_set))
    additional_info.fn2.sort()
    tmp_set = set(experiment_point_list.keys()) - set(labelled_point_dict.keys())
    additional_info.fp2.extend(list(tmp_set))
    additional_info.fp2.sort()
    error_info_template += r'<b>Baseline</b>: [{3:.2f}, {4:.2f}, {5:.2f}];  '
    error_info_template += r'<b>Experiment</b>: [{6:.2f}, {7:.2f}, {8:.2f}];  '
    error_info_template += r'<b>Error1</b>: x:{9:.2f}; y:{10:.2f}; z:{11:.2f}; L2:{12:.2f};'
    error_info_template += r'<b>Error2</b>: x:{13:.2f}; y:{14:.2f}; z:{15:.2f}; L2:{16:.2f};'
    error_info_template += r'<b>L2_difference</b>:{17:.2f}'
    error_summarys, benchmark, truly_positive_files = BaselineBenchmark(labelled_point_dict, baseline_point_list, experiment_point_list)

  document_text = r'"<h1>check predicted coordinates:</h1>"'
  document_text += "\n"

  for idx in range(len(truly_positive_files)):
    if usage_flag == 1:
      document_text = GenHtmlRowForLabelChecking(image_link_template, error_info_template, document_text, truly_positive_files,
                                  idx, point_dicts, image_folder, width)
      analysis_text = GenAnalysisText(num_data, truly_positive_files, additional_info, usage_flag)
    elif usage_flag == 2:
      document_text = GenHtmlRowForErrorAnalysis(image_link_template, error_info_template, document_text, truly_positive_files,
                                                 idx, point_dicts, image_folder, error_summary, width)
      analysis_text = GenAnalysisText(num_data, truly_positive_files, additional_info, usage_flag, error_summary)
    else:
      document_text = GenHtmlRowForBenchmark(image_link_template, error_info_template, document_text, truly_positive_files,
                                             idx, point_dicts, image_folder, error_summarys, benchmark, width)
      analysis_text = GenAnalysisText(num_data, truly_positive_files, additional_info, usage_flag, error_summarys, benchmark)

  abnormal_files = list(set(file_name_list) - set(truly_positive_files))
  abnormal_files.sort()
  for idx in range(len(abnormal_files)):
    document_text = GenHtmlRowOfAbnormalfiles(image_link_template, document_text, abnormal_files,
                                      idx, image_folder, usage_flag, width)

  html_report_path = os.path.join(output_folder, html_report_name)
  WriteToHtmlReportFile(document_text, analysis_text, html_report_path, width)


def GenAnalysisText(num_data, truly_positive_files, additional_info, usage_flag, error_summary=None,benchmark=None):
  analysis_text = "There are {0} cases in total,".format(num_data)
  analysis_text += "\n"
  analysis_text += r'<p style="color:blue;">{0} cases do not contain this landmark: {1}</p>'.format(
    len(additional_info['no_lm']), additional_info['no_lm'])
  analysis_text += "<p> </p>"

  if usage_flag == 2:
    analysis_text += r'<p style="color:red;">{0} false positives: {1}</p>'.format(len(additional_info['fp1']),
                                                                                  additional_info['fp1'])
    analysis_text += r'<p style="color:red;">{0} missings: {1}</p>'.format(len(additional_info['fn1']),
                                                                          additional_info['fn1'])
    analysis_text += r'<p>Error summary of the remaining {0} cases:</p>'.format(len(truly_positive_files))
    analysis_text += r'<p>mean:{0:.2f}</p>'.format(error_summary.mean_error)
    analysis_text += r'<p>median:{0:.2f}</p>'.format(error_summary.median_error)
    analysis_text += r'<p>max:{0:.2f}</p>'.format(error_summary.max_error)
    analysis_text += r'<p>min:{0:.2f}</p>'.format(error_summary.min_error)

  if usage_flag == 3:
    analysis_text += r'<p style="color:red;">{0} false positives of baseline: {1}</p>'.format(
      len(additional_info['fp1']), additional_info['fp1'])
    analysis_text += r'<p style="color:red;">{0} missings of baseline: {1}</p>'.format(len(additional_info['fn1']),
                                                                                       additional_info['fn1'])
    analysis_text += r'<p>Error summary of the remaining {0} cases of baseline:</p>'.format(len(truly_positive_files))
    analysis_text += r'<p>mean:{0:.2f}</p>'.format(error_summary[0].mean_error)
    analysis_text += r'<p>median:{0:.2f}</p>'.format(error_summary[0].median_error)
    analysis_text += r'<p>max:{0:.2f}</p>'.format(error_summary[0].max_error)
    analysis_text += r'<p>min:{0:.2f}</p>'.format(error_summary[0].min_error)

    analysis_text += r'<p style="color:red;">{0} false positives of experiment: {1}</p>'.format(
      len(additional_info['fp2']), additional_info['fp2'])
    analysis_text += r'<p style="color:red;">{0} missings of experiment: {1}</p>'.format(len(additional_info['fn2']),
                                                                                       additional_info['fn2'])
    analysis_text += r'<p>Error summary of the remaining {0} cases of experiment:</p>'.format(len(truly_positive_files))
    analysis_text += r'<p>mean:{0:.2f}</p>'.format(error_summary[1].mean_error)
    analysis_text += r'<p>median:{0:.2f}</p>'.format(error_summary[1].median_error)
    analysis_text += r'<p>max:{0:.2f}</p>'.format(error_summary[1].max_error)
    analysis_text += r'<p>min:{0:.2f}</p>'.format(error_summary[1].min_error)

    analysis_text += '<p style="color:red;">Change of each value(experiment to baseline):</p>'
    analysis_text += '<p>mean:{0:.2f}</p>'.format(benchmark.mean_diff)
    analysis_text += '<p>median:{0:.2f}</p>'.format(benchmark.median_diff)
    analysis_text += '<p>max:{0:.2f}</p>'.format(benchmark.max_diff)
    analysis_text += '<p>min:{0:.2f}</p>'.format(benchmark.min_diff)

  return analysis_text


def AddThreeImages(document_text, image_link_template, image_folder, images, width):
  for idx in range(3):
    document_text += "\n"
    image_info = r'<td>{0}</td>'.format(image_link_template.format(
      os.path.join(image_folder, images[idx]), width))
    document_text = AddDocumentText(document_text, image_info)
  return document_text


"""
Generate the text contents for abnormal cases, which contain the cases of unlabelled, false positives and miss detections. 
"""
def GenHtmlRowOfAbnormalfiles(image_link_template, document_text, abnormal_files, idx, image_folder, usage_flag, width):
  file_name = abnormal_files[idx]
  labelled_images = [file_name + '_labelled_axial.png',
                     file_name + '_labelled_coronal.png',
                     file_name + '_labelled_sagittal.png']
  detected_images = [file_name + '_detected_axial.png',
                     file_name + '_detected_coronal.png',
                     file_name + '_detected_sagittal.png']
  experiment_images = [file_name + '_experiment_axial.png',
                     file_name + '_experiment_coronal.png',
                     file_name + '_experiment_sagittal.png']
  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(idx, file_name)
  document_text = AddDocumentText(document_text, case_info)
  document_text += "\n"
  document_text = AddDocumentText(document_text, "<table border=1><tr>")
  document_text = AddThreeImages(document_text, image_link_template, image_folder, labelled_images, width)
  if usage_flag >= 2:
    document_text = AddThreeImages(document_text, image_link_template, image_folder, detected_images, width)
  if usage_flag == 3:
    document_text = AddThreeImages(document_text, image_link_template, image_folder, experiment_images, width)
  document_text += "\n"
  document_text = AddDocumentText(document_text, r'</tr></table>')
  return document_text


"""
Generate a line of html text contents for labelled cases, in the usage of label checking.
"""
def GenHtmlRowForLabelChecking(image_link_template, error_info_template, document_text, truly_positive_files,
                idx, point_dicts, image_folder, width):
  file_name = truly_positive_files[idx]
  labelled_point = list(point_dicts[0].values())[idx]
  labelled_images = [file_name + '_labelled_axial.png',
                     file_name + '_labelled_coronal.png',
                     file_name + '_labelled_sagittal.png']

  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(idx, file_name)
  error_info = error_info_template.format(labelled_point[0],
                                          labelled_point[1],
                                          labelled_point[2])
  document_text = AddDocumentText(document_text, case_info)
  document_text = AddDocumentText(document_text, error_info)
  document_text += "\n"
  document_text = AddDocumentText(document_text, "<table border=1><tr>")
  document_text = AddThreeImages(document_text, image_link_template, image_folder, labelled_images, width)
  document_text += "\n"
  document_text = AddDocumentText(document_text, r'</tr></table>')

  return document_text


"""
Generate a line of html text contents for truly positive files, in the usage of error analysis.
"""
def GenHtmlRowForErrorAnalysis(image_link_template, error_info_template, document_text, truly_positive_files,
                               idx, point_dicts, image_folder, error_summary, width):
  index = error_summary.sorted_index_list[idx]
  file_name = truly_positive_files[index]

  labelled_point = point_dicts[0][file_name]
  detected_point = point_dicts[1][file_name]

  labelled_images = [file_name + '_labelled_axial.png',
                     file_name + '_labelled_coronal.png',
                     file_name + '_labelled_sagittal.png']
  detected_images = [file_name + '_detected_axial.png',
                     file_name + '_detected_coronal.png',
                     file_name + '_detected_sagittal.png']

  x_error = error_summary.point_distance_list[index][0]
  y_error = error_summary.point_distance_list[index][1]
  z_error = error_summary.point_distance_list[index][2]
  l2_error = error_summary.l2_norm_error_list[index]

  error_info = error_info_template.format(labelled_point[0],
                                          labelled_point[1],
                                          labelled_point[2],
                                          detected_point[0],
                                          detected_point[1],
                                          detected_point[2],
                                          x_error,
                                          y_error,
                                          z_error,
                                          l2_error)

  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(index, file_name)
  document_text = AddDocumentText(document_text, case_info)
  document_text = AddDocumentText(document_text, error_info)
  document_text += "\n"
  document_text = AddDocumentText(document_text, "<table border=1><tr>")
  document_text = AddThreeImages(document_text, image_link_template, image_folder, labelled_images, width)
  document_text = AddThreeImages(document_text, image_link_template, image_folder, detected_images, width)
  document_text += "\n"
  document_text = AddDocumentText(document_text, r'</tr></table>')
  return document_text


"""
Generate a line of html text contents for truly positive files, in the usage of benchmark.
"""
def GenHtmlRowForBenchmark(image_link_template, error_info_template, document_text, truly_positive_files,
                           idx, point_dicts, image_folder, error_summarys, benchmark, width):
  index = benchmark.sorted_index_list[idx]
  file_name = truly_positive_files[index]

  labelled_point = point_dicts[0][file_name]
  detected_point = point_dicts[1][file_name]
  experiment_point = point_dicts[2][file_name]

  labelled_images = [file_name + '_labelled_axial.png',
                     file_name + '_labelled_coronal.png',
                     file_name + '_labelled_sagittal.png']
  baseline_images = [file_name + '_baseline_axial.png',
                     file_name + '_baseline_coronal.png',
                     file_name + '_baseline_sagittal.png']
  experiment_images = [file_name + '_experiment_axial.png',
                     file_name + '_experiment_coronal.png',
                     file_name + '_experiment_sagittal.png']

  x_error1 = error_summarys[0].point_distance_list[index][0]
  y_error1 = error_summarys[0].point_distance_list[index][1]
  z_error1 = error_summarys[0].point_distance_list[index][2]
  l2_error1 = error_summarys[0].l2_norm_error_list[index]
  x_error2 = error_summarys[1].point_distance_list[index][0]
  y_error2 = error_summarys[1].point_distance_list[index][1]
  z_error2 = error_summarys[1].point_distance_list[index][2]
  l2_error2 = error_summarys[1].l2_norm_error_list[index]

  l2_diff = benchmark.l2_norm_diff_list[index]

  error_info = error_info_template.format(labelled_point[0], labelled_point[1], labelled_point[2],
                                          detected_point[0], detected_point[1], detected_point[2],
                                          experiment_point[0], experiment_point[1], experiment_point[2],
                                          x_error1, y_error1, z_error1, l2_error1,
                                          x_error2, y_error2, z_error2, l2_error2,
                                          l2_diff)
  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(index, file_name)
  document_text = AddDocumentText(document_text, case_info)
  document_text = AddDocumentText(document_text, error_info)
  document_text += "\n"
  document_text = AddDocumentText(document_text, "<table border=1><tr>")
  document_text = AddThreeImages(document_text, image_link_template, image_folder, labelled_images, width)
  document_text = AddThreeImages(document_text, image_link_template, image_folder, baseline_images, width)
  document_text = AddThreeImages(document_text, image_link_template, image_folder, experiment_images, width)
  document_text += "\n"
  document_text = AddDocumentText(document_text, r'</tr></table>')
  return document_text


def gen_html_report(landmarks_list, usage_flag, output_folder):
  """
  Generate landmark evaluation HTML report.
  Input arguments:
    file_name_list:   The list of 3D volume file names including postfix.
    point_dicts:      A list of dicts containing the labelled(detected,experiment) points' coordinates.
    additional_info:  A dict containing the information of no_landmark, false positives
                      and miss detections.
    usage_flag:       A integer indicating the usage of the html tool.
                      1 for ground truth only
                      2 for error analysis
                      3 for benchmark comparison
    output_folder:    The output folder containing the html report.
    image_folder:     The image folder containing the captured 2D plane images.
                      Must be a relative path to the output_folder.
    html_report_name: The HTML report file name.
  """
  labelled_landmarks = landmarks_list[0]

  if usage_flag == 2:
    detected_landmarks = landmarks_list[1]
    assert len(labelled_landmarks.keys()) == len(detected_landmarks.keys())

  image_list = list(labelled_landmarks.keys())
  for landmark_name in labelled_landmarks[image_list[0]].keys():
    print("Generating html report for landmark {}.".format(landmark_name))
    image_link_template = r"<div class='content'><img border=0  src= '{0}'  hspace=1  width={1} class='pic'></div>"
    error_info_template = r'<b>Labelled</b>: [{0:.2f}, {1:.2f}, {2:.2f}];'
    document_text = r'"<h1>check predicted coordinates:</h1>"'
    document_text += "\n"

    for image_idx, image_name in enumerate(image_list):
      landmark_world = labelled_landmarks[image_name][landmark_name]
      if usage_flag == 1:
        document_text = gen_html_row_for_label_checking(image_link_template, error_info_template, document_text, image_list,
                                    image_idx, landmark_name, landmark_world, picture_folder='./pictures', width=200)
        analysis_text = gen_analysis_text(len(image_list))
      
      elif usage_flag == 2:
        document_text = gen_html_row_for_label_checking(image_link_template, error_info_template, document_text, image_list,
                                    image_idx, landmark_name, landmark_world, picture_folder='./pictures', width=200)
        analysis_text = gen_analysis_text(len(image_list))

      else:
        raise ValueError('Undefined usage flag!')
  
    html_report_name = 'result_analysis.html'.format(landmark_name)
    html_report_path = os.path.join(output_folder, 'lm{}'.format(landmark_name), html_report_name)
    WriteToHtmlReportFile(document_text, analysis_text, html_report_path, width=200)


def gen_analysis_text(num_data):
  analysis_text = "There are {0} cases in total,".format(num_data)
  analysis_text += "\n"

  return analysis_text


def gen_html_row_for_label_checking(image_link_template, error_info_template, document_text, image_list,
                                    image_idx, landmark_name, landmark_world, picture_folder, width):
  """
  Generate a line of html text contents for labelled cases, in the usage of label checking.
  """
  image_name = image_list[image_idx]
  image_basename = image_name.split('/')[0]
  labelled_images = [image_basename + '_label_lm{}_axial.png'.format(landmark_name),
                     image_basename + '_label_lm{}_coronal.png'.format(landmark_name),
                     image_basename + '_label_lm{}_sagittal.png'.format(landmark_name)]

  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(image_idx, image_name)
  error_info = error_info_template.format(landmark_world[0],
                                          landmark_world[1],
                                          landmark_world[2])
  document_text = AddDocumentText(document_text, case_info)
  document_text = AddDocumentText(document_text, error_info)
  document_text += "\n"
  document_text = AddDocumentText(document_text, "<table border=1><tr>")
  document_text = AddThreeImages(document_text, image_link_template, picture_folder, labelled_images, width)
  document_text += "\n"
  document_text = AddDocumentText(document_text, r'</tr></table>')

  return document_text