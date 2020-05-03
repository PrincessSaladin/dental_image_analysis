import os
import pandas as pd

from segmentation3d.utils.file_io import readlines
from segmentation3d.core.seg_infer import read_test_txt


def read_train_txt(imlist_file):
    """ read single-modality txt file
    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths
    """
    lines = readlines(imlist_file)
    num_cases = int(lines[0])

    if len(lines)-1 < num_cases * 2:
        raise ValueError('too few lines in imlist file')

    im_list, seg_list = [], []
    for i in range(num_cases):
        im_path, seg_path = lines[1 + i * 2], lines[2 + i * 2]
        im_list.append([im_path])
        seg_list.append(seg_path)

    return im_list, seg_list



def split_dataset():

    batch_idx = 1
    spacing = str(2.0)

    # get all image names
    segmentation_dataset_folder = '/mnt/projects/CT_Dental/dataset/segmentation'
    train_list_file = os.path.join(segmentation_dataset_folder, 'train_server.txt')
    _, seg_path = read_train_txt(train_list_file)
    training_image_names = []
    for path in seg_path:
        image_name = os.path.basename(os.path.dirname(path))
        training_image_names.append(image_name)

    # get image list of images that have landmark file
    image_names_with_landmark = []
    landmark_file_folder = '/mnt/projects/CT_Dental/landmark'
    landmark_files = os.listdir(landmark_file_folder)
    for landmark_file in landmark_files:
        if landmark_file.endswith('.csv'):
            image_names_with_landmark.append(landmark_file.split('.')[0])

    type = 'server'
    # folder_root = '/mnt/projects/CT_Dental'
    folder_root = '/shenlab/lab_stor6/projects/CT_Dental'
    save_folder = '/mnt/projects/CT_Dental/dataset/landmark_detection'
    server_image_folder = os.path.join(folder_root, 'data')
    server_landmark_folder = os.path.join(folder_root, 'landmark')
    server_landmark_mask_folder = \
        os.path.join(folder_root, 'landmark_mask/batch_{}_{}mm'.format(batch_idx, spacing))

    # generate dataset for the training
    content = []
    training_names = list(set(training_image_names) & set(image_names_with_landmark))
    training_names.sort()
    for name in training_names:
        image_path = os.path.join(server_image_folder, name, 'org.mha')
        mask_path = os.path.join(server_image_folder, name, 'seg.mha')
        landmark_file_path = os.path.join(server_landmark_folder, '{}.csv'.format(name))
        landmark_mask_path = os.path.join(server_landmark_mask_folder, '{}.mha'.format(name))
        content.append([name, image_path, mask_path, landmark_file_path, landmark_mask_path])

    csv_file_path = os.path.join(save_folder, 'train_{}_{}.csv'.format(batch_idx, type))
    columns = ['image_name', 'image_path', 'organ_mask_path', 'landmark_file_path', 'landmark_mask_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)

    # generate dataset for test
    content = []
    test_names = list(set(image_names_with_landmark) - set(training_image_names))
    test_names.sort()
    for name in test_names:
        image_path = os.path.join(server_image_folder, name, 'org.mha')
        mask_path = os.path.join(server_image_folder, name, 'seg.mha')
        landmark_file_path = os.path.join(server_landmark_folder, '{}.csv'.format(name))
        landmark_mask_path = os.path.join(server_landmark_mask_folder, '{}.mha'.format(name))
        content.append([name, image_path, mask_path, landmark_file_path, landmark_mask_path])

    csv_file_path = os.path.join(save_folder, 'test_{}_{}.csv'.format(batch_idx, type))
    columns = ['image_name', 'image_path', 'organ_mask_path', 'landmark_file_path', 'landmark_mask_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':

    steps = [1]

    if 1 in steps:
        split_dataset()


