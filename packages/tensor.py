import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class tensor_data:
    def __init__(self):
        self.session_dir = None
        self.model = None
        self.batch_size = None
        self.img_height = None
        self.img_width = None
        self.class_names = None
        self.train_ds = None
        self.val_ds = None
        self.history = None

    def build_tensor(self, file_name):
        global tensor
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
                file_name,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
                file_name,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        # Model shape should always be the same
        self.model = Sequential([
                layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(len(self.class_names))
            ])

        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
    def train_tensor(self):
        global tensor
        epochs=10
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
