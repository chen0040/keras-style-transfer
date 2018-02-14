import scipy
import scipy.io
import scipy.misc
from matplotlib.pyplot import imshow
from keras_style_transfer.library.style_transfer import StyleTransfer


def main():
    pretrained_model_dir_path = './pretrained-model'
    image_dir_path = './images'
    output_dir_path = './outputs'

    content_image = scipy.misc.imread(image_dir_path + "/louvre_small.jpg")
    imshow(content_image)

    style_image = scipy.misc.imread(image_dir_path + "/monet.jpg")
    imshow(style_image)

    vgg19_model_path = pretrained_model_dir_path + "/imagenet-vgg-verydeep-19.mat"
    ss = StyleTransfer(vgg19_model_path)

    generated_image = ss.fit_and_transform(content_image, style_image,
                                           output_dir_path=output_dir_path)



if __name__ == '__main__':
    main()
