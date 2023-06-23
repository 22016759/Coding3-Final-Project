# Coding3-Final-Project

## Project overview

The goal of this project was to use CycleGAN (cyclic generative Adversarial Network) to transform the style of the characters Tom and Jerry from the animated series Tom and Jerry. I tried to convert the image of the Tom into Jerry's style, and I also tried to convert the image of Jerry into the style of the Tom. The code is primarily based on Tensorflow and Tensorflow's Pix2Pix example.

CycleGAN is a model for image-to-image conversion that is able to learn the transformation without paired training samples. This means that we can convert one type of image (such as the image of a Tom) into another type of image (such as the image of Jerry), even if we don't have one-to-one correspondence between the image of a Tom and the image of Jerry. In the readme, I show how to use CycleGAN for style transformations, such as converting Tom to Jerry.

<img width="947" alt="截屏2023-06-23 19 05 51" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2a564cc8-c02a-4902-a8b0-f2801c2e3152">

## Data loading and preprocessing

The training of the data set went through three iterations. The dataset used in the first code was the Tom and Jerry dataset that I searched and downloaded in Kaggle. Data set download link：https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

```
train_Tom = tf.data.Dataset.list_files('/content/tom_and_jerry3/trainA/tom/*.jpg')
train_Jerry = tf.data.Dataset.list_files('/content/tom_and_jerry3/trainB/jerry/*.jpg')

test_Tom = tf.data.Dataset.list_files('/content/tom_and_jerry3/trainA/tom/*.jpg')
test_Jerry = tf.data.Dataset.list_files('/content/tom_and_jerry3/trainB/jerry/*.jpg')

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image
```
The code starts by loading an image dataset from Google Drive and dividing it up.
In random shake, the picture is resized to 286 x 286 and then randomly cropped to 256 x 256.
In a random mirror, pictures flip randomly from left to right.

### You can see the sample image loaded as follows:

 <img width="500" alt="截屏2023-06-23 22 10 42" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2a18b5f9-c840-4977-b67b-81b89d6f5cd1"><img width="500" alt="截屏2023-06-23 22 10 53" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/abb4a9f2-9853-4aaf-bda1-8877bae8a4e7">

## Definition model

This model consists of four parts: two generators (Generator_G and Generator_F) and two discriminators (Discriminator_X and Discriminator_Y). 

### The key code is as follows:

```
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

```
This code uses the Pix2Pix implementation in Tensorflow to create the generator and discriminator. The generator is used to perform image transformations (for example, from Tom to Jerry), while the discriminator is used to determine whether the image is real (for example, whether an image really looks like Jerry).

<p align="center">
 <img width="674" alt="截屏2023-06-22 22 39 54" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/370830dd-d36b-4152-b3c8-11a693a5208a">

 <img width="666" alt="截屏2023-06-23 22 19 52" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/701b5294-aa95-4ce4-b3a9-a6ea21907b75">
</p>

## Training model

Adam optimizer is used to train the model, and three loss functions are set up: generator loss, discriminator loss and cyclic consistency loss. 

### The key code is as follows:

```
def train_step(real_x, real_y):
    # persistent is set to True because gen_tape and disc_tape is used more than once to calculate the gradients.
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
      
        ...
      
        # Compute losses
        gen_g_loss = generator_loss(disc_real_y, fake_y, real_y)
        gen_f_loss = generator_loss(disc_real_x, fake_x, real_x)
      
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
        
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
      
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_loss = disc_x_loss + disc_y_loss

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = gen_tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
    generator_f_gradients = gen_tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)
  
    ...
  
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))
    ...
```




