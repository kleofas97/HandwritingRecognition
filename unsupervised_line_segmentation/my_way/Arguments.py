import argparse
from typing import NamedTuple
import os


class Arguments(NamedTuple):
    CUDA_nb: str
    input_shape: int
    train_set_path: str
    train_set_size: int
    validation_set_path: str
    validation_set_size: int
    learning_rate: float
    epochs: int
    batch_size: int
    train: bool
    continue_from_best: bool
    test_img_path: str
    path_to_model: str
    # version: int


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--CUDA_nb', dest='CUDA_nb', type=str, help='Number of CUDA devices',
                        default='0', required=False)
    parser.add_argument('--input_shape', dest='input_shape', help='Patch size of input images',
                        default=130, required=False)
    parser.add_argument('--train_set_path', dest='train_set_path', type=str,
                        help='Path to the directory containing images', required=False,
                        default=r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\normalized\data\train')
    parser.add_argument('--train_set_size', dest='train_set_size', type=str,
                        help='number of patches to be prepared', default=30000, required=False)
    parser.add_argument('--validation_set_path', dest='validation_set_path', type=str,
                        help='Path to the directory containing images for testing', required=False,
                        default=r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\normalized\data\val')
    parser.add_argument('--validation_set_size', dest='validation_set_size', type=int,
                        help='number of patches to be prepared for validation', required=False,
                        default=300)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        help='Starting learning rate', default=0.001, required=False)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, required=False)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, required=False)
    parser.add_argument('--train', dest='train', type=bool,
                        help='If model is to be trained or just to predict', default=True,
                        required=False)
    parser.add_argument('--continue_from_best', dest='continue_from_best',
                        help='Continue training previous model', type=bool, default=False,
                        required=False)
    parser.add_argument('--test_img_path', dest='test_img_path',
                        help='path to test image (train == false needed)', type=str,
                        required=False)
    parser.add_argument('--path_to_model', dest='path_to_model', type=str,
                        help='path to model to be loaded (only if continue_from_best is true',
                        required=False, default=str(os.path.join(os.getcwd(), 'learned_model')))
    # parser.add_argument('--version',dest='version',type=int, help='version_number')

    return Arguments(**vars(parser.parse_args()))
