import cv2


def sliding_window(image, step, window_size):
    """

    :param image: image
    :param step: integer
    :param window_size: (width, height)
    :return: x ,y, image
    """
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


def resize(image, width):
    ratio = width / image.shape[1]
    width = int(width)
    height = int(image.shape[0] * ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_pyramid(image, scale, minsize):
    yield image
    while True:
        width = image.shape[1] // scale
        image = resize(image, width)
        if image.shape[0] < minsize[1] or image.shape[1] < minsize[0]:
            break
        yield image
