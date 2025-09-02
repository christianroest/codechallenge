import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model


def build_model(input_shape: tuple):
    """
    Basic model class that implements a very simple model to test the rest of the pipeline

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

        Returns:
        tensorflow.keras.models.Model: Keras model.
    """
    i = Input(input_shape)
    o = Conv2D(3, (3, 3), padding="same")(i)
    return Model(i, o)


if __name__ == "__main__":
    in_shape = 1024, 1024, 3
    m = build_model(in_shape)
    m.summary()
