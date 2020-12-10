import unsupervised_line_segmentation.my_way.model as model_op
import unsupervised_line_segmentation.my_way.Arguments as arguments
import keras
import os
import numpy as np
import unsupervised_line_segmentation.my_way.preprocessing.my_pairs as pairs


Args = arguments.parse_args()
if Args.prepare_dataset:
    pairs.prepare_dataset_on_disk(dataset_path_train=Args.data_train_dir,
                                  dataset_path_val=Args.data_val_dir,
                                  path_to_output=Args.dataset_path,
                                  train_set_size=Args.train_set_size,
                                  val_set_size=Args.validation_set_size,
                                  patch_size=Args.input_shape)
nb_of_samples_train = len(os.listdir(os.path.join(Args.data_train_dir, 'train_0')))
nb_of_samples_val = len(os.path.join(Args.data_val_dir, 'val_0'))
train_steps = np.floor(nb_of_samples_train / Args.batch_size)
val_steps = np.floor(nb_of_samples_val / Args.batch_size)
input_shape = (Args.input_shape, Args.input_shape, 1)

if Args.train == True:
    # prepare data
    generator_train = model_op.genereate_batch(path_1=os.path.join(Args.data_train_dir, 'train_0'),
                                               path_2=os.path.join(Args.data_train_dir, 'train_1'),
                                               batch_size=Args.batch_size)
    generator_val = model_op.genereate_batch(path_1=os.path.join(Args.data_val_dir, 'val_0'),
                                             path_2=os.path.join(Args.data_val_dir, 'val_1'),
                                             batch_size=Args.batch_size)
    if Args.continue_from_best is True:
        assert Args.path_to_model is not None, "invalid path to model"
        model = keras.models.load_model(Args.path_to_model)
    else:
        model = model_op.built_model(input_shape=input_shape)
    model, history = model_op.fit_model(model=model, generator_train=generator_train,
                                        path_to_model=Args.path_to_model,
                                        generator_val=generator_val,
                                        epochs=Args.epochs,
                                        learning_rate=Args.learning_rate, train_steps=train_steps,
                                        val_steps=val_steps, patch_size=Args.input_shape)
else:
    model = keras.models.load_model(Args.path_to_model)
    # TODO prepare way to prepare image as input
# PART II WORD RECOGNITION - AT THIS TIME WE SHOULD HAVE READY CUTTED IMAGES WITH SEGMENTED TEXT LINES
    prediction = model.predict(Args.test_img_path)
    # TODO cut images to lines for next DNN
