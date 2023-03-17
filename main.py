import shutil

from PIL import Image, ImageOps, ImageDraw, ImageFilter
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def resize_and_padding(image, target_size):
    width, height = image.size

    scale = min(target_size[1] / height, target_size[0] / width)

    resize_width = int(width * scale)
    resize_height = int(height * scale)

    pil_image = image.resize((resize_width, resize_height))
    padding_bottom, padding_right = target_size[1] - resize_height, target_size[0] - resize_width
    pil_image = ImageOps.expand(pil_image, (0, 0, padding_right, padding_bottom), fill=0)
    return pil_image, (padding_bottom, padding_right), (height, width)


def draw_line(image, color_range=(-5, 5), width_range=(7, 12), count_range=(1, 2), radius_range=(1, 2)):
    np_image = np.asarray(image)
    median_color = np.median(np_image[np_image > 0])
    black_ratio = np.sum(np_image == 0) / (np_image.shape[0] * np_image.shape[1])
    while True:
        pil_image = Image.fromarray(np.zeros(np_image.shape, dtype=np_image.dtype))
        draw_image = ImageDraw.Draw(pil_image)
        for _ in range(random.randint(count_range[0], count_range[1])):
            color = int(np.clip(median_color + random.uniform(color_range[0], color_range[1]), 0, 255))
            draw_image.line([(random.randint(0, image.width), random.randint(0, image.height)),
                             (random.randint(0, image.width), random.randint(0, image.height))],
                            fill=color,
                            width=random.randint(width_range[0], width_range[1]))
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.randint(radius_range[0], radius_range[1])))
        draw_np_image = np.asarray(pil_image)
        if random.randint(0, 1) > black_ratio:
            merge_np_image = np.maximum(np_image, draw_np_image)
        else:
            merge_np_image = np.minimum(np_image, 255 - draw_np_image)
        if np.sum(np.abs(np_image - merge_np_image)) > 0:
            break
    return Image.fromarray(merge_np_image)


def draw_point(image, ellipse_size_range=(10, 25), color_range=(-5, 5), width_range=(1, 4), count_range=(1, 3),
               radius_range=(1, 2)):
    np_image = np.asarray(image)
    median_color = np.median(np_image[np_image > 0])
    black_ratio = np.sum(np_image == 0) / (np_image.shape[0] * np_image.shape[1])

    while True:
        pil_image = Image.fromarray(np.zeros(np_image.shape, dtype=np_image.dtype))
        draw_image = ImageDraw.Draw(pil_image)
        for _ in range(random.randint(count_range[0], count_range[1])):
            color = int(np.clip(median_color + random.uniform(color_range[0], color_range[1]), 0, 255))
            ellipse_width, ellipse_height = random.randint(ellipse_size_range[0],
                                                           ellipse_size_range[1]), random.randint(ellipse_size_range[0],
                                                                                                  ellipse_size_range[1])
            start_x, start_y = random.randint(0, image.width - ellipse_width), random.randint(0,
                                                                                              image.height - ellipse_height)
            draw_image.ellipse([(start_x, start_y), (start_x + ellipse_width, start_y + ellipse_height)],
                               fill=color,
                               width=random.randint(width_range[0], width_range[1]))
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.randint(radius_range[0], radius_range[1])))
        draw_np_image = np.asarray(pil_image)
        if random.randint(0, 1) > black_ratio:
            merge_np_image = np.maximum(np_image, draw_np_image)
        else:
            merge_np_image = np.minimum(np_image, 255 - draw_np_image)
        if np.sum(np.abs(np_image - merge_np_image)) > 0:
            break
    return Image.fromarray(merge_np_image)


def draw_mix(image):
    line_image = draw_line(image)
    point_line_image = draw_point(line_image)
    return point_line_image


def get_mask(pil_image, process_pil_image):
    np_image = np.asarray(pil_image)
    process_np_image = np.asarray(process_pil_image)
    diff_image = np.abs(np_image - process_np_image)
    diff_image[diff_image > 0] = 255
    diff_pil_image = Image.fromarray(diff_image)
    return diff_pil_image


def write(output_sub_dir_path, sample_num, anomaly_ratio, image_max_size, image_min_size, x, y):
    output_good_sub_dir_path = os.path.join(output_sub_dir_path, 'raw', 'good')
    os.makedirs(output_good_sub_dir_path, exist_ok=True)
    for file_index in tqdm(range(int(sample_num * (1. - anomaly_ratio))), desc=f'write at {output_sub_dir_path}'):
        mnist_index = random.randint(0, y.shape[0] - 1)
        pil_image = Image.fromarray(x[mnist_index])
        pil_image, _, _ = resize_and_padding(pil_image, (
            random.randint(image_min_size, image_max_size), random.randint(image_min_size, image_max_size)))
        pil_image.save(os.path.join(output_good_sub_dir_path, f'{file_index:04d}_{mnist_index:02d}.png'))

    process_dict = {
        "line": draw_line,
        "point": draw_point,
        "mix": draw_mix
    }
    for file_index in tqdm(range(int(sample_num * (anomaly_ratio))), desc=f'write at {output_sub_dir_path}'):
        mnist_index = random.randint(0, y.shape[0] - 1)
        pil_image = Image.fromarray(x[mnist_index])
        pil_image, _, _ = resize_and_padding(pil_image, (
            random.randint(image_min_size, image_max_size), random.randint(image_min_size, image_max_size)))

        process_key = random.choice(list(process_dict.keys()))
        process_pil_image = process_dict[process_key](pil_image)
        output_anomaly_sub_key_raw_dir_path = os.path.join(output_sub_dir_path, 'raw', process_key)
        os.makedirs(output_anomaly_sub_key_raw_dir_path, exist_ok=True)
        process_pil_image.save(
            os.path.join(output_anomaly_sub_key_raw_dir_path, f'{file_index:04d}_{mnist_index:02d}.png'))

        output_anomaly_sub_key_gt_dir_path = os.path.join(output_sub_dir_path, 'ground_truth', process_key)
        os.makedirs(output_anomaly_sub_key_gt_dir_path, exist_ok=True)
        diff_pil_image = get_mask(pil_image, process_pil_image)
        diff_pil_image.save(os.path.join(output_anomaly_sub_key_gt_dir_path, f'{file_index:04d}_{mnist_index:02d}.png'))


def main(output_dir_path, target_digit, train_sample_num, train_anomaly_ratio, valid_sample_num, valid_anomaly_ratio, image_max_size,
         image_min_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train_indexes = y_train == target_digit
    x_train, y_train = x_train[y_train_indexes], y_train[y_train_indexes]
    y_test_indexes = y_test == target_digit
    x_test, y_test = x_test[y_test_indexes], y_test[y_test_indexes]
    output_train_dir_path = os.path.join(output_dir_path, 'train')
    write(output_train_dir_path, train_sample_num, train_anomaly_ratio, image_max_size, image_min_size, x_train,
          y_train)

    output_train_dir_path = os.path.join(output_dir_path, 'train_valid')
    write(output_train_dir_path, valid_sample_num, valid_anomaly_ratio, image_max_size, image_min_size, x_train,
          y_train)

    output_valid_dir_path = os.path.join(output_dir_path, 'valid')
    write(output_valid_dir_path, valid_sample_num, valid_anomaly_ratio, image_max_size, image_min_size, x_test, y_test)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset')
    parser.add_argument('--target_digit', type=int, default=5)
    parser.add_argument('--train_sample_num', type=int, default=12500)
    parser.add_argument('--train_anomaly_ratio', type=float, default=0.2)
    parser.add_argument('--valid_sample_num', type=int, default=100)
    parser.add_argument('--valid_anomaly_ratio', type=float, default=0.8)
    parser.add_argument('--image_max_size', type=int, default=136)
    parser.add_argument('--image_min_size', type=int, default=120)
    args = parser.parse_args()

    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
