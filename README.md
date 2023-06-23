# Coding3-Final-Project

## Project overview

The goal of this project was to use CycleGAN (cyclic generative Adversarial Network) to transform the style of the characters Tom and Jerry from the animated series Tom and Jerry. I tried to convert the image of the Tom into Jerry's style, and I also tried to convert the image of Jerry into the style of the Tom. The code is primarily based on Tensorflow and Tensorflow's Pix2Pix example.

CycleGAN is a model for image-to-image conversion that is able to learn the transformation without paired training samples. This means that we can convert one type of image (such as the image of a Tom) into another type of image (such as the image of Jerry), even if we don't have one-to-one correspondence between the image of a Tom and the image of Jerry. In the readme, I show how to use CycleGAN for style transformations, such as converting Tom to Jerry.

<img width="947" alt="截屏2023-06-23 19 05 51" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2a564cc8-c02a-4902-a8b0-f2801c2e3152">

## Data loading and preprocessing

The training of the data set went through three iterations. The dataset used in the first code was the Tom and Jerry dataset that I searched and downloaded in Kaggle. 

Data set download link：https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

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

<p align="center">
 <img width="500" alt="截屏2023-06-23 22 10 42" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2a18b5f9-c840-4977-b67b-81b89d6f5cd1">
 
 <img width="500" alt="截屏2023-06-23 22 10 53" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/abb4a9f2-9853-4aaf-bda1-8877bae8a4e7">
 </p>

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

## Train and save the model

During training, I generate and save images according to a certain number of steps in order to visualize the progress of the model. 

### The key code is as follows:

```
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n+=1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model is visible.
    generate_images(generator_g, sample_horse, epoch)
```

In this code, I train each pair of images in the data set. For every 10 sessions, we print a dot to show our progress. At the end of each epoch, we use a consistent sample image to generate and save an image so that we can see the improvement of the model over time.

### Training process：

<img width="972" alt="截屏2023-06-22 23 06 13" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/a9181aa3-9564-4d04-bcff-61b4f4acf85c">

<img width="972" alt="截屏2023-06-22 23 09 55" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/34443430-707e-4cd1-9cb9-a384beebe0d8">

<img width="968" alt="截屏2023-06-23 00 03 19" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/f6ebf20e-3207-4cf8-9166-a659cc96042f">

## Generate and display images

```
def generate_images(model, test_input, epoch):
    prediction = model(test_input)
        
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], tf.squeeze(prediction[0])]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

In this code, I use the model to make a prediction on the input image, and then display the input image and the prediction image. This allows you to visually see how the model is performing. I save the generated images for subsequent analysis and comparison.

<p align="center">
<img width="500" alt="截屏2023-06-22 18 10 30" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2822ac57-b37e-4269-9b6c-63deb391806b">

<img width="509" alt="截屏2023-06-22 21 03 49" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/bb5b6c82-b892-4f28-84c3-222994582ac8">
</p>

The results obtained from the previous training are the same as the results shown in the above figure. The changes in the trained pictures are not perfect, only a small part of the colors have changed, but it can be seen that they are gradually changing into Jerry's colors.
In view of this situation, I thought that the original data set might not be fully matched in this training task, so I re-cropped and adjusted the positions and sizes of Tom and Jerry in the data set, so that the main characters were concentrated in the middle of the screen, and re-adjusted the data set.

### The result is the following picture

<img width="947" alt="截屏2023-06-23 22 50 18" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/8df9cea7-c8fe-4b9a-a16f-3b02530df767">

![下载 (2)](https://github.com/22016759/Coding3-Final-Project/assets/119021236/d07d966c-9efd-4d4d-8bd3-8f950b691faa)
![下载 (3)](https://github.com/22016759/Coding3-Final-Project/assets/119021236/d23b6343-6f93-495b-9302-a75bd24b0a51)
![下载 (4)](https://github.com/22016759/Coding3-Final-Project/assets/119021236/a98a818e-7014-4aae-bec5-1f37ec862500)

## Problems and deficiencies

1. Fixed input image size: The current model is only suitable for 256x256 images and may require model modification or preprocessing steps for images of different sizes.

2. Single sample evaluation: During training, we only show the prediction results of one sample image, which may not fully reflect the performance of the model. A better approach might be to display a prediction of a set of sample images.

3. Lack of early stopping mechanism: The code does not implement early stopping of the model, that is, stopping training when the model does not improve on the validation set. This can cause the model to overfit and over-learn from the training data, resulting in poor performance on new, unseen data.

4. Parameter tuning is not flexible enough: In the code, many important parameters, such as the number of training rounds (epochs), the learning rate of the optimizer, etc., are hard-coded in the code, which is often inconvenient in practice.

## To improve these problems in the future, the following ways can be considered:

1. Increase input flexibility: Modify the preprocessing steps and model so that it can handle images of various sizes.

2. Enhanced model evaluation: During training, more variables can be used to evaluate the performance of the model, and more complex evaluation measures such as adversarial loss and cycle consistency loss can be considered.

3. Add early stop mechanism: Using early stop prevents overfitting and can reduce training time to some extent. You can set up a validation set and monitor its performance on that set. When performance stops improving, stop training and save the best model.

4. Parameter tuning: To make your code more usable, consider including some key parameters (such as learning rate, number of training rounds, etc.) as function parameters or command line parameters that can be changed without modifying your code.

## Citation
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```
Code borrows from pix2pix and DCGAN. The data loader is modified from DCGAN and Context-Encoder. The generative network is adopted from neural-style with Instance Normalization.


