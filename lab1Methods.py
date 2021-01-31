from PIL import Image
import math

def get_pixel(image, x, y):
    x = bound(x, 0, image['width'] - 1)
    y = bound(y, 0, image['height'] - 1)
    return image['pixels'][x + image['width'] * y]


def bound(n, lower, upper):
    return min(max(n, lower), upper)  # Because if statements are too mainstream


def set_pixel(image, x, y, c):
    image['pixels'][x + image['width'] * y] = c


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][::],
    }
    for x in range(image['width']):
        for y in range(image['height']):
            newColor = func(get_pixel(image, x, y))
            set_pixel(result, x, y, newColor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255 - c)


# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE:
    kernel = {"size": n, "multipliers": n x n list}
    """
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * len(image['pixels']),
    }
    for x in range(image['width']):
        for y in range(image['height']):
            set_pixel(result, x, y, getTransformedPixel(image, kernel, x, y))
    return result


def getTransformedPixel(image, kernel, x, y):
    c = 0
    size = round(len(kernel) ** 0.5)
    for kx in range(0, size):
        for ky in range(0, size):
            k = kernel[kx + size * ky]
            c += k * get_pixel(image, x + kx - size // 2, y + ky - size // 2)
    return c


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    return apply_per_pixel(image, lambda c : round(bound(c, 0, 255)))


# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    k = create_blur_kernel(n)

    # then compute the correlation of the input image with that kernel
    newImage = correlate(image, k)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    return round_and_clip_image(newImage)


def create_blur_kernel(n):
    return [1 / (n ** 2)] * (n ** 2)


def sharpened(image, n):
    """
    Return a new image representing the result of applying an unsharp mask (with
    kernel size n) to the given input image. The mask is created by building a
    negative blur kernel of size n, then adding a positive weight of 2 to the
    kernel's center. This kernel is then applied to the image using a correlation.
    """
    k = [-1 / (n ** 2)] * (n ** 2)  # Create negative blur kernel
    k[(n ** 2) // 2] += 2  # Add a positive 2 weight to the blur kernel
    newImage = correlate(image, k)  # Apply kernel
    return round_and_clip_image(newImage)  # Return a valid image


def edges(image):
    """
    Return a new image representing the result of applying a sobel operator to the given input image.
    Two images, one corresponding to horizontal edges, and the other corresponding to vertical, are
    created by correlating the image with the horizontal and vertical sobel operators, respectively.
    The results are then combined to produce a final, edge-emphasized, picture.
    """
    kx = [-1, 0, 1,
          -2, 0, 2,
          -1, 0, 1]
    ky = [-1, -2, -1,
          0, 0, 0,
          1, 2, 1]
    ox = correlate(image, kx)
    oy = correlate(image, ky)
    o = {
        'width': image['width'],
        'height': image['height'],
        'pixels': [0] * len(image['pixels']),
    }
    for i in range(len(o['pixels'])):
        o['pixels'][i] = math.sqrt(ox['pixels'][i] ** 2 + oy['pixels'][i] ** 2)
    return round_and_clip_image(o)


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES
def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()