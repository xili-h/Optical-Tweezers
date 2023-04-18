import numpy as np

import deeptrack as dt
from deeptrack.extras import datasets
import matplotlib.pyplot as plt

#datasets.load('QuantumDots')
#IMAGE_SIZE_x = 1080
#IMAGE_SIZE_y = 19
IMAGE_SIZE_x = 128
IMAGE_SIZE_y = 128

#particle = dt.PointParticle(
#    position=lambda: [np.random.rand() * IMAGE_SIZE_x, np.random.rand() * IMAGE_SIZE_y],
#    z=lambda: np.random.randn() * 5,
#    intensity=lambda: 1 + np.random.rand() * 9,
#    position_unit="pixel",
#)
particle = dt.scatterers.Sphere(
    position=lambda: [np.random.rand() * IMAGE_SIZE_x, np.random.rand() * IMAGE_SIZE_y],
    z=lambda: -200 + np.random.randn() * 50,
    radius=lambda: np.random.uniform(0.9, 1.1) * 1e-6,
    value=lambda: np.random.uniform(1.37, 1.42),
    position_unit="pixel"
)

number_of_particles = lambda: np.random.randint(1, 2)

particles = particle ^ number_of_particles

# optics = dt.Fluorescence(
#     NA=lambda: 0.6 + np.random.rand() * 0.2,
#     wavelength=500e-9,
#     resolution=1e-6,
#     magnification=10,
#     output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
# )

objective = dt.optics.Brightfield(
    wavelength=450e-9,
    NA=0.65,
    resolution=4e-6, 
    magnification=70,
    refractive_index_medium=1.333,
    output_region=(0, 0, IMAGE_SIZE_x, IMAGE_SIZE_y)
)

normalization = dt.math.NormalizeMinMax(
    min=lambda: np.random.rand() * 0.4,
    max=lambda min: min + 0.1 + np.random.rand() * 0.5,
)

noise = dt.noises.Poisson(
    snr=lambda: 20 + np.random.rand() * 3,
    background=normalization.min
)

imaged_particle = objective(particles)

dataset = imaged_particle >> normalization >> noise

CIRCLE_RADIUS = 15

X, Y = np.mgrid[:2*CIRCLE_RADIUS, :2*CIRCLE_RADIUS]

circle = (X - CIRCLE_RADIUS + 0.5)**2 + (Y - CIRCLE_RADIUS + 0.5)**2 < CIRCLE_RADIUS**2
circle = np.expand_dims(circle, axis=-1)

get_masks = dt.SampleToMasks(
    lambda: lambda image: circle,
    output_region=objective.output_region,
    merge_method="or"
)

def get_label(image_of_particles):
    return get_masks.update().resolve(image_of_particles)

# def get_label(image_of_particles):
#     label = np.zeros((*image_of_particles.shape[:2], 3))
    
#     X, Y = np.meshgrid(
#         np.arange(0, image_of_particles.shape[0]), 
#         np.arange(0, image_of_particles.shape[1])
#     )

#     for property in image_of_particles.properties:
#         if "position" in property:
#             position = property["position"]
#             distance_map = (X - position[1])**2 + (Y - position[0])**2
#             label[distance_map < 9] = 1
            
#     #label[..., 0] = 1 - np.max(label[..., 1:], axis=-1)
    
#     return label

#WARNING!!!!: DO NOT SHOW PLOT WHEN TRAINNING
if False:
    NUMBER_OF_IMAGES = 10

    for _ in range(NUMBER_OF_IMAGES):
        plt.figure(figsize=(15, 5))
        dataset.update()
        image_of_particle = dataset.resolve(skip_augmentations=True)
        particle_label = get_label(image_of_particle)
        plt.subplot(1, 2, 1)
        plt.imshow(image_of_particle[..., 0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(particle_label[..., 0]*1.0, cmap="gray")
        plt.show()
else:
    #define the model
    loss = dt.losses.flatten(
        dt.losses.weighted_crossentropy((10, 1))
    )
    metric = dt.losses.flatten(
        dt.losses.weighted_crossentropy((1, 1))
    )
    model = dt.models.UNet(
        (None, None, 1), 
        conv_layers_dimensions=[16, 32, 64],
        base_conv_layers_dimensions=[128, 128], 
        loss=loss,
        metrics=[metric],
        output_activation="sigmoid"
    )

    #model.summary()

    #train the model
    TRAIN_MODEL = True

    validation_set_size = 100

    validation_set = [dataset.update().resolve() for _ in range(validation_set_size)]
    validation_labels = [get_label(image) for image in validation_set]

    if TRAIN_MODEL:
        generator = dt.generators.ContinuousGenerator(
            dataset & (dataset >> get_label),
            batch_size=16,
            min_data_size=2e3,
            max_data_size=1e4,
        )

        with generator:

            # Train 5 epochs with weighted loss
            h = model.fit(generator,
                        epochs=20,
                            validation_data=(
                                np.array(validation_set),
                                np.array(validation_labels)
                            ))

            model.compile(loss=metric, optimizer="adam")

            # Train 30 epochs with unweighted loss
            h2 = model.fit(generator,
                        epochs=60,
                        validation_data=(
                            np.array(validation_set),
                            np.array(validation_labels)
                        ))
            
    else:
        model_path = datasets.load_model("QuantumDots")
        model.load_weights(model_path)

    model.save_weights("Winter/model_newtrain.h5")




