import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model


def build_model(
    input_shape: tuple,
    num_classes: int,
) -> tf.keras.models.Model:
    """
    Basic model class that implements a very simple model to test the rest of the pipeline

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

        Returns:
        tensorflow.keras.models.Model: Keras model.
    """
    i = Input(input_shape)

    # Add a layer that rescales input images to the [0, 1] range
    i = tf.keras.layers.Rescaling(1.0 / 255)(i)

    o = Conv2D(num_classes, (3, 3), padding="same")(i)
    return Model(i, o)


if __name__ == "__main__":
    in_shape = 1024, 1024, 3
    m = build_model(in_shape, 1)
    m.summary()
