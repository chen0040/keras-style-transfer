import scipy
import scipy.io
import scipy.misc
from matplotlib.pyplot import imshow
from keras_style_transfer.library.style_transfer import StyleTransfer


def main():
    pretrained_model_dir_path = '../training/pretrained-model'
    image_dir_path = '../training/images'

    content_image = scipy.misc.imread(image_dir_path + "/louvre.jpg")
    imshow(content_image)

    style_image = scipy.misc.imread(image_dir_path + "/monet_800600.jpg")
    imshow(style_image)

    ss = StyleTransfer()
    ss.load_vgg19_model(pretrained_model_dir_path)
    generated_image = ss.generate_noise_image(content_image)
    imshow(generated_image[0])




if __name__ == '__main__':
    main()
