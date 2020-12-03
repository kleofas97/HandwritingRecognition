import unsupervised_line_segmentation.my_way.model as model_op
import unsupervised_line_segmentation.my_way.Arguments as arguments
import keras
import os
import numpy as np
import unsupervised_line_segmentation.my_way.preprocessing.my_pairs as pairs

TRAIN_PATH_0 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\grayscale_dataset_prepared_128px\train\train_0'
TRAIN_PATH_1 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\grayscale_dataset_prepared_128px\train\train_1'
VAL_PATH_0 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\grayscale_dataset_prepared_128px\val\val_0'
VAL_PATH_1 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\grayscale_dataset_prepared_128px\val\val_1'
Args = arguments.parse_args()
# PREPARING DATASET (USE IF WE DON HAVE PATCHES READY FOR LEARNING
# TO USE THIS PART YOU MUST HAVE A FOLDER WITH DATA SET PREPARED AND IN IT SPLITTED IMAGES FOR TRAIN AND VAL
# BY DEFAULT I HAVE SPLITTED THEM BY 0.9 AND 0.1.
# TO SPLIT THE DATASET, IF YOU HAVE ONLY ONE FOLDER WITH ALL PICTURES USE SPLIT_DATASET.PY, BUT MAKE
# SURE TO MANUALLY SPLIT ALL IMAGES TO TWO FOLDERS WITH THE SAME (OR MOSTLY) NUMBER OF PICUTRES
dataset_path_train = os.path.join(os.path.dirname(os.getcwd()), "my_way", 'grayscale_dataset',
                                  'train')
dataset_path_val = os.path.join(os.path.dirname(os.getcwd()), "my_way", 'grayscale_dataset', 'val')
output_path = os.path.join(os.path.dirname(os.getcwd()), "my_way",
                           'grayscale_dataset_prepared_150px', )
pairs.prepare_dataset(dataset_path_train=dataset_path_train, dataset_path_val=dataset_path_val,
                      path_to_output=output_path, train_set_size=30000, val_set_size=3000,
                      patch_size=Args.input_shape)
# END OF DATASET PREPARATION


nb_of_samples_train = len(os.listdir(TRAIN_PATH_0))
nb_of_samples_val = len(os.listdir(VAL_PATH_0))
train_steps = np.floor(nb_of_samples_train / Args.batch_size)
val_steps = np.floor(nb_of_samples_val / Args.batch_size)
input_shape = (Args.input_shape, Args.input_shape, 1)

if Args.train == True:
    # prepare data
    generator_train = model_op.genereate_batch(TRAIN_PATH_0, TRAIN_PATH_1,
                                               batch_size=Args.batch_size)
    generator_val = model_op.genereate_batch(VAL_PATH_0, VAL_PATH_1, batch_size=Args.batch_size)
    if Args.continue_from_best is True:
        assert Args.path_to_model is not None, "invalid path to model"
        model = keras.models.load_model(Args.path_to_model)
    else:
        model = model_op.built_model(input_shape=input_shape)
    model, history = model_op.fit_model(model=model, generator_train=generator_train,
                                        path_to_model=Args.path_to_model,
                                        generator_val=generator_val,
                                        epochs=Args.epochs,
                                        learning_rate=0.01, train_steps=train_steps,
                                        val_steps=val_steps)
else:
    model = keras.models.load_model(Args.path_to_model)
    # TODO prepare way to prepare image as input
    prediction = model.predict(Args.test_img_path)
    # TODO cut images to lines for next DNN
