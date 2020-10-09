from tensorflow.keras import layers, models

# 32C5S2 means a convolution layer with 32 feature maps using a 5x5 filter and stride 2
# P2 means max pooling using 2x2 filter and stride 2
# F256 means fully connected dense layer with 256 units

def CNN_32C5S1_P2_64C5S1_P2_F256(input_dim, output_dim):

    input_1 = layers.Input(shape=(input_dim[0],input_dim[1],input_dim[2]))
    x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(input_1)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2,2)(x)
    out_feat = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(out_feat)
    output_class = layers.Dense(output_dim, activation='softmax')(x)

    model = models.Model(inputs=[input_1], outputs=[output_class])

    return model

def CNN_32C5S1_P2_64C5S1_P2_F256_F2_out_F2(input_dim, output_dim):

    input_1 = layers.Input(shape=(input_dim[0],input_dim[1],input_dim[2]))
    x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(input_1)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2,2)(x)
    out_feat = layers.Flatten()(x)

    #x = layers.Dense(256, activation='relu')(out_feat)
    x = layers.Dense(2, activation='linear')(out_feat)
    output_class = layers.Dense(output_dim, activation='softmax')(x)

    model = models.Model(inputs=[input_1], outputs=[output_class, x])

    return model

def CNN_32C5S1_P2_64C5S1_P2_F256_F8_out_F8(input_dim, output_dim):

    input_1 = layers.Input(shape=(input_dim[0],input_dim[1],input_dim[2]))
    x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(input_1)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2,2)(x)
    out_feat = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(out_feat)
    x = layers.Dense(8, activation='relu')(x)
    output_class = layers.Dense(output_dim, activation='softmax')(x)

    model = models.Model(inputs=[input_1], outputs=[output_class, x])

    return model