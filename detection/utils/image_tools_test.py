import SimpleITK as sitk

from detection.utils.image_tools import mask_to_mesh


def test_mask_to_mesh():
    mask_path_mha = '/mnt/projects/CT_Dental/test_seg.mha'
    mask_path_nii = '/mnt/projects/CT_Dental/test_seg.nii.gz'

    stl_path = '/mnt/projects/CT_Dental/test_seg.stl'
    label = 2
    mask_to_mesh(mask_path_mha, stl_path, label)


if __name__ == '__main__':

    test_mask_to_mesh()

