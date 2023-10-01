import tensorflow as tf

def flip_image(image, label):
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image, label

def rotate_image(image, label):
    rotated_image = tf.image.rot90(image)
    return rotated_image, label

def random_crop_image(image, label):
    cropped_image = tf.image.random_crop(image, size=image.shape)
    return cropped_image, label

def adjust_brightness_image(image, label):
    brightness_adjusted_image = tf.image.adjust_brightness(image, delta=0.2)
    return brightness_adjusted_image, label

def adjust_contrast_image(image, label):
    contrast_adjusted_image = tf.image.adjust_contrast(image, contrast_factor=2.0)
    return contrast_adjusted_image, label