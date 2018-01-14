import scipy
import scipy.io
import scipy.misc
from matplotlib.pyplot import imshow
from keras_style_transfer.library.style_transfer import StyleTransfer


def main():
    pretrained_model_dir_path = '../training/pretrained-model'
    image_dir_path = '../training/images'
    output_dir_path = '../training/outputs'

    content_image = scipy.misc.imread(image_dir_path + "/louvre_small.jpg")
    imshow(content_image)

    style_image = scipy.misc.imread(image_dir_path + "/monet.jpg")
    imshow(style_image)

    ss = StyleTransfer()
    ss.load_vgg19_model(pretrained_model_dir_path)

    ss.fit(content_image, style_image, output_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
