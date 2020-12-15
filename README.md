# ImageTransformer
This notebook shows a basic implementation of a transformer (decoder) architecture for image generation in TensorFlow 2.

It demonstrates how to use a transformer decoder to learn a generative representation of the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset and perform an autoregressive image reconstruction.

Mnist dataset examples:

![MNIST examples](https://github.com/GregorKobsik/ImageTransformer/blob/main/imgs/samples_mnist.png)

To reduce the number of color values, we perform a color quantization, e.g. we compute k-means clustering to get 8 color clusters and thus reduce our color palette.

Quantized examples:

![MNIST quantized](https://github.com/GregorKobsik/ImageTransformer/blob/main/imgs/quantized_mnist.png)

Afterwards we serialize the images to obtain linear sequences of length 784 per image, which can be fed into the model as used in NLP. 

See the notebook to get an in-depth explanation of the model.

## Results

We perform image reconstruction, e.g. we take mnist images, remove the bottom half of the image, quantize it and let our model reconstruct the missing part. Afterwards we can revert the quantization and obtain a new generated mnist image. Compare the output and the input to see that the model does not memorize the inputs but creates new images.

Input data:

![mnist input](https://github.com/GregorKobsik/ImageTransformer/blob/main/imgs/input_mnist.png)

Bottom half removed:

![mnist top half](https://github.com/GregorKobsik/ImageTransformer/blob/main/imgs/input_mnist_half.png)

Generated output:

![model output](https://github.com/GregorKobsik/ImageTransformer/blob/main/imgs/output_mnist.png)

## Further reading

- [Transformers Tutorial](https://www.tensorflow.org/tutorials/text/transformer) - In depth tutorial on transformers in TF2.

- [Illustrated Transformers Guide](http://jalammar.github.io/illustrated-transformer/) - Quick and intuitive explanation of transformers.

- [Image GPT Blog](https://openai.com/blog/image-gpt/) - original ImageGPT by Chen et al.

- [ImageGPT in PyTorch](https://github.com/teddykoker/image-gpt) - an implementation of ImageGPT for PyTorch
