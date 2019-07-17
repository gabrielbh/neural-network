from skimage.color import rgb2gray
from scipy.misc import imread
import numpy as np
import random
from tensorflow.keras import layers, models, optimizers
from scipy import ndimage
import matplotlib.pyplot as plt

from . import sol5_utils

LARGER_CROP_PARAM = 3
KERNEL_SIZE = (3, 3)
GREY_LEVEL_MAX_VAL = 256
GREY_SCALE = 1
SUBTRACT_VALUE = 0.5

def read_image(filename, representation):
  """
  function which reads an image file and converts it into a given representation.
  This function returns an image, normalized to the range [0, 1].
  """
  im = imread(filename).astype(np.float64) / (GREY_LEVEL_MAX_VAL - 1)
  if (representation == 1):
      im_g = rgb2gray(im)
      return im_g
  return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Returns a generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, height, width, 1),
     target_batch is made of clean images, and source_batch is their respective randomly corrupted version
     according to corruption_func(im).
    """
    pictures = {}
    height, width = crop_size
    while True:
        source_batch = []
        target_batch = []

        for _ in range(batch_size):
            rand_im = np.random.choice(filenames)
            if rand_im not in pictures:
                im = read_image(rand_im, GREY_SCALE)
                pictures[rand_im] = im
            else:
                im = pictures[rand_im]
            rand_row = random.randint(0, im.shape[0] - LARGER_CROP_PARAM * height)
            rand_col = random.randint(0, im.shape[1] - LARGER_CROP_PARAM * width)
            corrupted_im = corruption_func(im[rand_row: rand_row + LARGER_CROP_PARAM * height,
                                           rand_col: rand_col + LARGER_CROP_PARAM * height])

            rand_patch_row = random.randint(0, height * (LARGER_CROP_PARAM - 1))
            rand_patch_col = random.randint(0, width * (LARGER_CROP_PARAM - 1))
            source_im = corrupted_im[rand_patch_row: rand_patch_row + height, rand_patch_col: rand_patch_col + width]
            target_im = im[rand_row + rand_patch_row: rand_row + rand_patch_row + height,
            rand_patch_col + rand_col: rand_patch_col + rand_col + width]

            source_batch.append(source_im.reshape(crop_size[0], crop_size[1], 1)  - SUBTRACT_VALUE)
            target_batch.append(target_im.reshape(crop_size[0], crop_size[1], 1) - SUBTRACT_VALUE)

        yield np.asarray(source_batch), np.asarray(target_batch)


def resblock(input_tensor, num_channels):
    """
    Function that takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration
    """
    first_conv = layers.Conv2D(num_channels, KERNEL_SIZE, padding='same')(input_tensor)
    relu = layers.Activation('relu')(first_conv)
    second_conv = layers.Conv2D(num_channels, KERNEL_SIZE, padding='same')(relu)
    add = layers.Add()([input_tensor, second_conv])
    return layers.Activation('relu')(add)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    return an untrained Keras model, with input dimension the shape of (height, width, 1),
    and all convolutional layers with number of output channels equal to num_channels,
    except the very last convolutional layer which should have a single output channel.
    """
    the_input = layers.Input((height, width, GREY_SCALE))
    first_conv = layers.Conv2D(num_channels, KERNEL_SIZE, padding='same')(the_input)
    residual_block_input = layers.Activation('relu')(first_conv)
    for loop in range(num_res_blocks):
        residual_block_input = resblock(residual_block_input, num_channels)
    last_conv = layers.Conv2D(GREY_SCALE, KERNEL_SIZE, padding='same')(residual_block_input)
    the_output = layers.Add()([the_input, last_conv])
    return models.Model(inputs=the_input, outputs=the_output)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Function that trains the dataset.
    """
    training_set = images[:int(0.8 * len(images))]
    validation_set = images[int(0.8 * len(images)):]
    crop_size = [model.input_shape[1], model.input_shape[2]]
    load_training = load_dataset(training_set, batch_size, corruption_func, crop_size)
    load_validation = load_dataset(validation_set, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(beta_2=0.9))
    model.fit_generator(load_training, validation_data=load_validation, validation_steps=num_valid_samples,
                        epochs=num_epochs, steps_per_epoch=steps_per_epoch)


def restore_image(corrupted_image, base_model):
    height, width = corrupted_image.shape
    input_tensor = layers.Input((height, width, GREY_SCALE))
    apply_base_model = base_model(input_tensor)
    new_model = models.Model(inputs=input_tensor, outputs=apply_base_model)
    corrupted_image -= 0.5
    predict = new_model.predict(corrupted_image.reshape(1, height, width, GREY_SCALE))
    fit_predict_size = predict.reshape((height, width)).astype(np.float64)
    fit_predict_range = fit_predict_size + 0.5
    clip_predict = np.clip(fit_predict_range, 0, 1)
    return clip_predict


def add_gaussian_noise(image, min_sigma, max_sigma):
    sigma = random.uniform(min_sigma, max_sigma)
    mean = 0
    add_gaussian = np.random.normal(mean, sigma, image.shape)
    image += add_gaussian
    round_pic = np.divide(np.round(np.multiply(image, GREY_LEVEL_MAX_VAL - 1)), GREY_LEVEL_MAX_VAL - 1)
    cliPic = np.clip(round_pic, 0, 1)
    return cliPic


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    images_paths_list = sol5_utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)

    def corruption_func(pic):
        return add_gaussian_noise(pic, 0, 0.2)

    if quick_mode:
        train_model(model, images_paths_list, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images_paths_list, corruption_func, 100, 100, 5, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    angle_in_radians = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return ndimage.filters.convolve(image, angle_in_radians)


def random_motion_blur(image, list_of_kernel_sizes):
    random_angle = random.uniform(0, np.pi)
    random_kernel = random.choice(list_of_kernel_sizes)
    motion_blur = add_motion_blur(image, random_kernel, random_angle)
    round_pic = np.divide(np.round(np.multiply(motion_blur, GREY_LEVEL_MAX_VAL - 1)), GREY_LEVEL_MAX_VAL - 1)
    cliPic = np.clip(round_pic, 0, 1)
    return cliPic


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    images_paths_list = sol5_utils.images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)

    def corruption_func(pic):
        return random_motion_blur(pic, [7])

    if quick_mode:
        train_model(model, images_paths_list, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images_paths_list, corruption_func, 100, 100, 10, 1000)
    return model