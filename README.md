# Coding3-Final-Project

## Project overview

The goal of this project was to use CycleGAN (cyclic generative Adversarial Network) to transform the style of the characters Tom and Jerry from the animated series Tom and Jerry. I tried to convert the image of the Tom into Jerry's style, and I also tried to convert the image of Jerry into the style of the Tom. The code is primarily based on Tensorflow and Tensorflow's Pix2Pix example.

CycleGAN is a model for image-to-image conversion that is able to learn the transformation without paired training samples. This means that we can convert one type of image (such as the image of a Tom) into another type of image (such as the image of Jerry), even if we don't have one-to-one correspondence between the image of a Tom and the image of Jerry. In the readme, I show how to use CycleGAN for style transformations, such as converting Tom to Jerry.

<img width="947" alt="截屏2023-06-23 19 05 51" src="https://github.com/22016759/Coding3-Final-Project/assets/119021236/2a564cc8-c02a-4902-a8b0-f2801c2e3152">

## Data loading and preprocessing

1. The training of the data set went through three iterations. The dataset used in the first code was the Tom and Jerry dataset that I searched and downloaded in Kaggle. Data set download link：https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

### The code starts by loading an image dataset from Google Drive and dividing it up.

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

2.
