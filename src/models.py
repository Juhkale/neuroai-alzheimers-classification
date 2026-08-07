import tensorflow as tf

def build_baseline_cnn(input_shape=(176, 176, 1), num_classes=4):
    model = tf.keras.Sequential([
        # Block 1: first conv layer "sees" the raw image.
        # 32 filters, 3x3 kernel size, 'relu' activation, padding='same'
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # Block 2: deeper features, more filters (64)
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # Block 3: even deeper, more filters (128)
        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # Global average pooling instead of Flatten -- drastically reduces
        # parameter count compared to flattening, which helps prevent
        # overfitting on a dataset this size.
        tf.keras.layers.GlobalAveragePooling2D(),

        # A dense layer to combine features, then Dropout to reduce overfitting
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer: one unit per class, softmax turns outputs into probabilities
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model