import tensorflow as tf
import os

from espcn import ESPCN
import prepare_data
import util

TEST_IMAGE_DIR = './test_images'
TEST_RESULT_DIR = './result'


def rebuild(img_name):
    """
    图像超分辨率重建
    :return:
    """
    lr_image, ori_image = prepare_data.preprocess_img(TEST_IMAGE_DIR+img_name)
    try:
        image_height, image_width, _ = lr_image.shape
    except:
        return
    with tf.Session() as sess:
        espcn = ESPCN(sess,
                      is_train=False,
                      image_height=image_height,
                      image_width=image_width,
                      image_channel=prepare_data.IMAGE_CHANNEl,
                      ratio=prepare_data.RATIO)
        sr_image = espcn.generate(lr_image / 255.0)

    
    # otherwise there would be error for image.save
    if not os.path.isdir(TEST_RESULT_DIR):
        os.makedirs(TEST_RESULT_DIR)

    # lr image
    # util.show_img_from_array(lr_image)
    util.save_img_from_array(
        lr_image, TEST_RESULT_DIR+img_name.split('.')[0]+'_lr.png')
    # original image
    # util.show_img_from_array(ori_image)
    util.save_img_from_array(ori_image, TEST_RESULT_DIR +
                             img_name.split('.')[0]+'_hr.png')
    # sr image
    # util.show_img_from_array(sr_image)
    util.save_img_from_array(sr_image, TEST_RESULT_DIR +
                             img_name.split('.')[0]+'_sr.png')


if __name__ == '__main__':
    img_list = [filename for filename in os.listdir(
        TEST_IMAGE_DIR) if filename.endswith('jpg')]
    for filename in img_list:
        rebuild(filename)
