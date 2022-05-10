import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Class to hold all the data for the model training
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

    # Build the model outline to train on
    def build_tensor(self, file_name, randomize):
        global tensor
        
        # Split up the data into training 80% and validation 20%
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

        # A hack to make randomization work on M1 Macbook Pro silicon.  Without
        # this tensorflow can't find a device to randomize images
        with tf.device('/CPU:0'):
            data_augmentation = keras.Sequential([
                    layers.RandomFlip("horizontal",
                        input_shape=(self.img_height,
                            self.img_width,
                            3)),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1)])

        self.class_names = self.train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        # Model shape should always be the same with randomizationa and without

        # If the user chose to randomize then it is added into the model
        if randomize:
            self.model = Sequential([
                    data_augmentation,
                    layers.Rescaling(1./255),
                    layers.Conv2D(16, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Dropout(0.2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(len(self.class_names))
                ])
        # No randomization, normally just going to make the training worse
        # since the model will trian for characteristics it shouldn't
        else:
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
    
    # Train the model on its data
    def train_tensor(self, epochs):
        global tensor
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
    
    # Run an image through the model and see what it thinks it is.  Only takes
    # in certain image data types, in this case PIL works
    def predict_tensor(self, img_data):

        img_array = tf.keras.utils.img_to_array(img_data)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return score
