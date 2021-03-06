import splitfolders
import os

PATH_TO_DATASET = os.path.join(os.path.dirname(os.getcwd()), 'cutted_pages')
PATH_TO_OUTPUT = os.path.join(os.path.dirname(os.getcwd()), 'normalized', 'data2')

splitfolders.ratio(input=PATH_TO_DATASET, output=PATH_TO_OUTPUT, ratio=(0.8, 0.2), seed=42)
