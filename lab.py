#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def newFilter(image):
        channels = [0, 0, 0]
        for i in range(3):
            channels[i] = filt({
                'height': image['height'],
                'width': image['width'],
                'pixels': [image['pixels'][j][i] for j in range(len(image['pixels']))]
            })
        a = channels[0]['pixels']
        b = channels[1]['pixels']
        c = channels[2]['pixels']
        return {
                'height': image['height'],
                'width': image['width'],
                'pixels': list(zip(*[c['pixels'] for c in channels]))
            }

    return newFilter


def make_blur_filter(n):
    return lambda image : blurred(image, n)


def make_sharpen_filter(n):
    return lambda image : sharpened(image, n)


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def newFilter(image):
        for f in filters:
            image = f(image)
        return image
    return newFilter


# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    getMap = filter_cascade([
        greyscale_image_from_color_image,
        compute_energy,
        cumulative_energy_map
    ])
    for i in range(ncols):
        print(str(i) + "/" + str(ncols))
        seam = minimum_energy_seam(getMap(image))
        image = image_without_seam(image, seam)
    return image


# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    return {
        'height': image['height'],
        'width': image['width'],
        'pixels': [round(px[0] * 0.299 + px[1] * 0.587 + px[2] * 0.114) for px in image['pixels']]
    }


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    map = {
        'height': energy['height'],
        'width': energy['width'],
        'pixels': energy['pixels'][:]
    }
    for r in range(1, energy['height']):
        for px in range(0, energy['width']):
            minEnergy = min (
                get_pixel(map, px + 1, r - 1),
                get_pixel(map, px, r - 1),
                get_pixel(map, px - 1, r - 1)
            )
            set_pixel(map, px, r, minEnergy + get_pixel(energy, px, r))
    return map


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """

    bottomRow = cem['pixels'][toLinear((0, cem['height'] - 1), cem['width']):]
    seam = [0] * cem['height']
    seam[0] = toLinear((bottomRow.index(min(bottomRow)), cem['height'] - 1), cem['width'])
    for i in range(0, cem['height'] - 1):
        prev = toCartesian(seam[i], cem['width'])
        minEnergy = get_pixel(cem, prev[0] + 1, prev[1] - 1)
        head = (min(prev[0] + 1, cem['width'] - 1), prev[1] - 1)
        for shift in range(2):
            energy = get_pixel(cem, prev[0] - shift, prev[1] - 1)
            if energy <= minEnergy:
                minEnergy = energy
                head = (max(prev[0] - shift, 0), prev[1] - 1)
        seam[i + 1] = toLinear(head, cem['width'])
    return seam


def toLinear(pos, w):
    return pos[0] + pos[1] * w


def toCartesian(pos, w):
    return pos % w, pos // w


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    newImage = {
        'height': image['height'],
        'width': image['width'] - 1,
        'pixels': image['pixels'][:]
    }
    for s in seam:
        del newImage['pixels'][s]
    return newImage


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


# LAB1 METHODS:
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
    return apply_per_pixel(image, lambda c: round(bound(c, 0, 255)))


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


def lagrangePoly(points):
    degree = len(points)
    ls = [0] * degree
    for j in range(degree):
        def lj(x, j=j):
            xj = points[j][0]
            product = 1
            for m in range(degree):
                if m != j:
                    xm = points[m][0]
                    product *= (x - xm)/(xj - xm)
            return product
        ls[j] = lj

    def L(x):
        sum = 0
        for j in range(len(ls)):
            xj = points[j][0]
            yj = points[j][1]
            sum += yj * ls[j](x)
        return sum
    return L


def colorCurve(r, g, b):

    def formatPoints(points):
        uniquePoints = {}
        for i in range(len(points)):
            old = bound(points[i][0], 0, 254)
            new = bound(points[i][1], 0, 254)
            uniquePoints[old] = new
        return [(0, 0)] + list(zip(list(uniquePoints.keys()), list(uniquePoints.values()))) + [(255, 255)]
    def roundPoly(points):
        return lambda x : round(lagrangePoly(points)(x))
    channelFilters = [roundPoly(formatPoints(r)), roundPoly(formatPoints(g)), roundPoly(formatPoints(b))]

    def applyCurve(image):
        channels = [0, 0, 0]
        for i in range(3):
            colorChannel = {
                'height': image['height'],
                'width': image['width'],
                'pixels': [image['pixels'][j][i] for j in range(len(image['pixels']))]
            }
            channels[i] = apply_per_pixel(colorChannel, channelFilters[i])
        return {
                'height': image['height'],
                'width': image['width'],
                'pixels': list(zip(*[c['pixels'] for c in channels]))
            }
    return applyCurve


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


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    """
    #Q1
    inverted_cat = color_filter_from_greyscale_filter(inverted)(load_color_image('test_images/cat.png'))
    save_color_image(inverted_cat, 'test_images/inverted_cat.png')

    #Q2
    blurred_python = color_filter_from_greyscale_filter(make_blur_filter(9))(load_color_image('test_images/python.png'))
    save_color_image(blurred_python, 'test_images/blurred_python.png')

    sharpened_sparrowchick = color_filter_from_greyscale_filter(make_sharpen_filter(7))(load_color_image('test_images/sparrowchick.png'))
    save_color_image(sharpened_sparrowchick, 'test_images/sharpened_sparrowchick.png')

    #Q3
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    frog = filt(load_color_image('test_images/frog.png'))
    save_color_image(frog, 'test_images/cascade_frog.png')

    #Q4
    twocats = load_color_image('test_images/twocats.png')
    save_color_image(seam_carving(twocats, 100), 'test_images/twocatsseam.png')
    """

    before = load_color_image('test_images/mushroom.png')
    save_color_image(colorCurve([(128, 0)], [(128, 100)], [(128, 100)])(before), 'test_images/colorcurvemushroom.png')
    Image.open('test_images/mushroom.png').show()
    Image.open('test_images/colorcurvemushroom.png').show()

    before = load_color_image('test_images/tree.png')
    save_color_image(colorCurve([(128, 80)], [(200, 60)], [(200, 30)])(before), 'test_images/colorcurvetree.png')
    Image.open('test_images/tree.png').show()
    Image.open('test_images/colorcurvetree.png').show()
