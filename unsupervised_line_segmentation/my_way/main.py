import unsupervised_line_segmentation.my_way.model as model_op
import unsupervised_line_segmentation.my_way.Arguments as arguments
import keras
import os
import numpy as np

TRAIN_PATH_0 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\train_2\image_0'
TRAIN_PATH_1 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\train_2\image_1'
VAL_PATH_0 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\val_2\image_0'
VAL_PATH_1 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\val_2\image_1'

Args = arguments.parse_args()
nb_of_samples_train = len(os.listdir(TRAIN_PATH_0))
nb_of_samples_val = len(os.listdir(VAL_PATH_0))
train_steps = np.ceil(nb_of_samples_train / Args.batch_size)
val_steps = np.ceil(nb_of_samples_val / Args.batch_size)
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
    # TODO prepare way to preapre image as input
    prediction = model.predict(Args.test_img_path)
    # TODO cut images to lines for next DNN
