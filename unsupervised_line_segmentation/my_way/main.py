import unsupervised_line_segmentation.my_way.model as model_op
import os
import unsupervised_line_segmentation.my_way.Arguments as arguments
import keras
import unsupervised_line_segmentation.my_way.preprocessing.my_pairs as my_pairs

Args = arguments.parse_args()
input_shape = (Args.input_shape, Args.input_shape, 1)


if Args.train == True:
    # prepare data
    # TODO prepare train and test data
    train_pair, train_label = my_pairs.unsupervised_loaddata(Args.train_set_path, Args.train_set_size,Args.input_shape)
    val_pair, val_label = my_pairs.unsupervised_loaddata(Args.validation_set_path, Args.validation_set_size,Args.input_shape)
    if Args.continue_from_best is True:
        assert Args.path_to_model is not None, "invalid path to model"
        model = keras.models.load_model(Args.path_to_model)
    else:
        model = model_op.built_model(input_shape=input_shape)
    model, history = model_op.fit_model(model, path_to_model=Args.path_to_model,
                                       train_pairs=train_pair,
                                        train_label=train_label,
                                        batch_size=Args.batch_size, epochs=Args.epochs,
                                        val_pairs=val_pair,
                                        val_label=val_label, learning_rate=Args.learning_rate)
else:
    model = keras.models.load_model(Args.path_to_model)
    #TODO prepare way to preapre image as input
    prediction = model.predict(Args.test_img_path)
    #TODO cut images to lines for next DNN
