from PIL import Image, ImageOps
import argparse
import os
import random
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


def write(output_sub_dir_path, sample_num, image_max_size, image_min_size, x):
    # train
    os.makedirs(output_sub_dir_path, exist_ok=True)
    for file_index in tqdm(range(sample_num), desc=f'write at {output_sub_dir_path}'):
        image = x[random.randint(0, x.shape[0]-2)]
        pil_image = Image.fromarray(image).convert('RGB')
        pil_image, _, _ = resize_and_padding(pil_image, (
            random.randint(image_min_size, image_max_size), random.randint(image_min_size, image_max_size)))
        pil_image.save(os.path.join(output_sub_dir_path, f'{file_index:08d}.png'))


def main(output_dir_path, target_digit, train_sample_num, valid_sample_num, image_max_size, image_min_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_target_train = x_train[y_train == target_digit]
    x_target_test = x_test[y_test == target_digit]
    x_etc_test = x_test[y_test != target_digit]

    output_train_dir_path = os.path.join(output_dir_path, 'train', 'good')
    write(output_train_dir_path, train_sample_num, image_max_size, image_min_size, x_target_train)

    output_valid_dir_path = os.path.join(output_dir_path, 'valid', 'good')
    write(output_valid_dir_path, valid_sample_num, image_max_size, image_min_size, x_target_test)

    output_valid_dir_path = os.path.join(output_dir_path, 'valid', 'anomaly')
    write(output_valid_dir_path, valid_sample_num, image_max_size, image_min_size, x_etc_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset')
    parser.add_argument('--target_digit', type=int, default=5)
    parser.add_argument('--train_sample_num', type=int, default=10)
    parser.add_argument('--valid_sample_num', type=int, default=10)
    parser.add_argument('--image_max_size', type=int, default=232)
    parser.add_argument('--image_min_size', type=int, default=216)
    args = parser.parse_args()

    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
