#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""

__all__ = (
    'generate_images',
)

import math
import os
import random
import exrex

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common
from common import OUTPUT_SHAPE

LOCALE_US = True

FONT_DIR = 'fonts'

PPI = 12 # points per inch
FONT_HEIGHT = int(2.5 * PPI)  # Pixel size to which the chars are resized

DETECT_OUTPUT_SHAPE = (64, 128)
READ_OUTPUT_SHAPE = (64, 128)

CHARS=common.CHARS[:]
CHARS.append(" ")

def make_character_images(font, output_height):
    font_size = output_height * 4
    font = ImageFont.truetype(font, font_size)
    height = max(font.getsize(d)[1] for d in CHARS)
    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf') or f.endswith('.otf')]
    for font in fonts:
        font_char_ims[font] = dict(make_character_images(os.path.join(folder_path, font), FONT_HEIGHT))
    return fonts, font_char_ims


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[c, 0., s],
                      [0., 1., 0.],
                      [-s, 0., c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[1., 0., 0.],
                      [0., c, -s],
                      [0., s, c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[c, -s, 0.],
                      [s, c, 0.],
                      [0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def get_transformed_shape_size(M, shape):
    """Return the size of the bounding box of a transformed shape."""
    h, w = shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    transformed_size = np.array(np.max(M * corners, axis=1) -
                                   np.min(M * corners, axis=1)).flatten()
    return transformed_size


def make_transform(yaw, pitch, roll, from_shape, bounds):
    """
    Make a 2x2 transform from the given parameters.
    :param yaw:
        Yaw angle to rotate by.
    :param pitch:
        Pitch angle to rotate by.
    :param roll:
        Roll angle to rotate by.
    :param from_shape:
        Shape of the image being tranformed.
    :param bounds:
        The scale will be selected such that the resulting shape's size is
        within these bounds.
    :return:
        A tuple `M`, `size` where `M` is the transformation, and `size` is the
        size of the bounding box containing the transformed shape.
    """
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    skewed_size = get_transformed_shape_size(M, from_shape)
    scale = np.min(np.array(bounds) / skewed_size)

    return M * scale, skewed_size * scale


def make_affine_transform(from_shape, to_shape,
                          yaw_range, pitch_range, roll_range, scale_range,
                          completely_inside=False):
    """
    Make a random affine transform for a shape, to fit within an output image.
    A rotation (specified in terms of yaw/pitch/roll) are selected based on
    ranges specified in the arguments.

    Scale is similarly specified by a range. A scale of 1.0 corresponds with an
    output image whose size equals that of the output image in exactly one
    dimension, with the other dimension being smaller.

    If `on_edge` is `False` translation is selected uniformly with the
    constraint that the transformed shape's bounding box lies entirely within
    the output shape. If `on_edge` is `True` translation is selected uniformly
    with the constraint that the transformed shape's bounding box intersects
    with one or more edges of the output shape.
    :param from_shape:
        The shape being transformed.
    :param to_shape:
        The shape of the output image.
    :param yaw_range:
        A (min, max) tuple defining the uniform distribution from which the yaw
        is selected.
    :param pitch_range:
        A (min, max) tuple defining the uniform distribution from which the
        pitch is selected.
    :param roll_range:
        A (min, max) tuple defining the uniform distribution from which the
        roll is selected.
    :param scale_range:
        A (min, max) tuple defining the uniform distribution from which the
        scale is selected. The maximum must be less than 1.0.
    :param completely_inside:
        Indicate whether the bounding box of the transformed shape's bounding
        box should lie entirely within the output shape (`True`) or whether
        only part of the part of the transformed shape's bounding box should
        lie within the output shape (`False`).
    :return:
        A tuple `M`, `out_of_bounds`, `scale` where `M` is the 2x3 affine
        transform described above, `out_of_bounds` indicates whether the
        transformed shape's bounding box partially lies outside of the output
        shape (note `out_of_bounds` is always `False` if `completely_inside` is
        `True`), and `scale` indicates the scale that was chosen.
    """
    yaw = random.uniform(*yaw_range)
    pitch = random.uniform(*pitch_range)
    roll = random.uniform(*roll_range)
    scale = random.uniform(*scale_range)
    bounds = scale * np.array([to_shape[1], to_shape[0]])

    M, transformed_size = make_transform(yaw, pitch, roll, from_shape, bounds)

    # Set `t` to the translation which puts the centre of the plate at 0, 0.
    t = M * np.matrix([[-from_shape[1], -from_shape[0]]]).T * 0.5

    # Determine out the x and y coordinates of the output shape centre.
    if completely_inside:
        x = random.uniform(transformed_size[0] / 2,
                           to_shape[1] - transformed_size[0] / 2)
        y = random.uniform(transformed_size[1] / 2,
                           to_shape[0] - transformed_size[1] / 2)
        out_of_bounds = False
    else:
        x = random.uniform(-transformed_size[0] / 2,
                           to_shape[1] + transformed_size[0] / 2)
        y = random.uniform(-transformed_size[1] / 2,
                           to_shape[0] + transformed_size[1] / 2)
        out_of_bounds = (x < transformed_size[0] / 2. or
                         x > to_shape[1] - transformed_size[0] / 2 or
                         y < transformed_size[1] / 2. or
                         y > to_shape[0] - transformed_size[1] / 2)

    # Add the above to `t` to get the final translation.
    t += np.matrix([[x], [y]])

    return np.hstack([M, t]), out_of_bounds, scale


def Xmake_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size) * 1.1

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2, 1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])
    return M, out_of_bounds

def generate_plate_number():
    # below is an imcomplete list of formats taken from wikipedia.
    # for now, ABC-123 is assumed to be the same as ABC 123 (but not ABC123) as the '-' if often not a '-' but
    # the silhouette of a state, bird, etc.
    # https://en.wikipedia.org/wiki/United_States_license_plate_designs_and_serial_formats
    formats = ['0[A-Z]{2}\d{4}',  # 0AB1234: Alabama
               '00[A-Z]{2}\d{3}',  # 00AB123: Alabama
               '\d{3} [A-Z]{3}',  # 123 ABC: Ark, Co, Conn, Fl, KS, KY, LA, MN, NM, ND, OR, WA, WI
               '[A-Z]{3} \d{3}',  # ABC 123: Alaska, Colorado, HI, IA, LA, MI, MS, NE, NM, NC, OK, OR, SC, VT
               '[A-Z]{3} \d{4}',  # ABC 1234: NC, OH, PA, TX, WI, NY, VT, MI
               '[A-Z]{3}\d{3}',  # ABC123: IN, MO
               '[A-Z]{3}\d{4}',  # ABC1234: AZ, WA
               '\d{1}[A-Z]{3}\d{3}',  # 1ABC234: California
               '[A-Z]{2}\d{5}',  #AB12345: Connecticut
               '\d{1}[A-Z]{2} [A-Z]{2}\d{1}',  #1AB CD2: Connecticut
               '\d{1}[A-Z]{2} [A-Z]{2}\d{1}',  # 1ABCD2: Connecticut
               '\d{6}',  # 123456: DE, RI
               '[A-Z]{2} \d{4}',  # AB 1234: District of Columbia, Guam
               '[A-Z]{3} [A-Z]\d{2}',  #ABC D12: Florida
               '[A-Z]\d{2} \d[A-Z]{2}',  #A12 3BC: Florida
               '\d{3} \d[A-Z]{2}',  #123 4AB:
               '[A-Z] \d{6}',  #A 123456: ID
               '0[A-Z] \d{5}',  #0A 12345: ID
               '0[A-Z] [A-Z]\d{4}',  #0A B1234: ID
               '0[A-Z] [A-Z]{2}\d{3}',  #0A BC123: ID -- there are more . . .
               '[A-Z]{2} \d{5}',  #AB 12345: IL
               '[A-Z]{2}\d \d{4}',  #AB1 2345: IL
               '\d{3}[A-Z]{1,3}',  #123A, 123AB, 123ABC: IN
               '\d{4} [A-Z]{2}',  # 1234 AB: ME
               '\d{4}[A-Z]{2}',  # 1234AB: ME, SC
               '\d[A-Z]{2}\d{4}',  #1AB2345: MA
               '\d[A-Z]{2} [A-Z]\d{2}',  #1AB C23: MI
               '[A-Z]{2}\d [A-Z]\d[A-Z]',  #AB1 C2D: MO
               '\d{3} [A-Z]\d{2}',  #123 A45: NV
               '\d{3} \d{3,4}',  #123 456; 123 4567: NH
               ]
    format = random.choice(formats)
    return exrex.getone(format)


def rounded_rect(shape, radius):
    out = np.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):

    def generate_code():
        return "{}{}{}{} {}{}{}".format(
            random.choice(common.LETTERS),
            random.choice(common.LETTERS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.LETTERS),
            random.choice(common.LETTERS),
            random.choice(common.LETTERS))

    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()

    text_mask = np.zeros(out_shape)

    x = h_padding
    y = v_padding
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (np.ones(out_shape) * plate_color * (1. - text_mask) +
             np.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_US_plate(font_height, character_images, plate_number=None):

    plate_number = plate_number if plate_number else generate_plate_number()
    plate_shape = (6 * PPI, 12 * PPI) # 6" high x 12" wide

    right_left_padding = 0.25 * PPI
    # top_bottom_padding = 1.25 * PPI
    plate_number_fits = False
    tries = 0
    while not plate_number_fits:

        spacing = (0.25 + random.uniform(-0.2, -0.05)) * PPI

        text_width = sum(character_images[c].shape[1] for c in plate_number)
        text_width += (len(plate_number) - 1) * spacing
        text_height = character_images['A'].shape[0]  # height of the characters
        # make sure the text fits on the plate
        if text_width + 2 * right_left_padding <= plate_shape[1]:
            plate_number_fits = True
        else:
            tries = tries + 1
            if tries > 10:
                return None, None, None

    radius = 1 + int(font_height * 0.1 * random.random())

    text_color, plate_color = pick_colors()
    
    text_mask = np.zeros(plate_shape)

    # starting x,y so that plate number is centered on the plate
    x = (plate_shape[1] - text_width)/2
    y = (plate_shape[0] - text_height)/2
    for c in plate_number:
        char_im = character_images[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (np.ones(plate_shape) * plate_color * (1. - text_mask) +
             np.ones(plate_shape) * text_color * text_mask)

    return plate, rounded_rect(plate_shape, radius), plate_number.replace(" ", "s")


def generate_background(num_bg_images, output_shape):

    found = False
    while not found:
        fname = "background_images/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))

        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= output_shape[1] and
                    bg.shape[0] >= output_shape[0]):
            found = True

    x = random.randint(0, bg.shape[1] - output_shape[1])
    y = random.randint(0, bg.shape[0] - output_shape[0])
    bg = bg[y:y + output_shape[0], x:x + output_shape[1]]

    return bg


def generate_detect_im(character_images, number_background_images, output_shape):
    output_size = np.array([output_shape[1], output_shape[0]])

    bg = generate_background(number_background_images, output_shape)

    plate, plate_mask, code = generate_US_plate(FONT_HEIGHT, character_images) if LOCALE_US else generate_plate(FONT_HEIGHT, character_images)
    if plate is None:
        return None, None, None
    plate_shape = plate.shape
    plate_size = np.array([[plate.shape[1], plate.shape[0]]]).T

    M, out_of_bounds, scale = make_affine_transform(plate.shape,
                                                    output_shape,
                                                    roll_range=(-0.3, 0.3),
                                                    pitch_range=(-0.2, 0.2),
                                                    yaw_range=(-1.2, 1.2),
                                                    scale_range=(0.3, 1.0),
                                                    completely_inside=False)

    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (output_shape[1], output_shape[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)

    plate_centre = np.array(M * np.concatenate([plate_size * 0.5,
                                                      [[1.]]])).T[0]
    plate_centre = plate_centre / output_size
    skewed_size = get_transformed_shape_size(M[:, :2], plate_shape)
    scale = np.max(skewed_size / output_size)

    return out, plate_centre, scale


def generate_im(character_images, number_background_images, output_shape):
    bg = generate_background(number_background_images, output_shape)

    plate, plate_mask, code = generate_US_plate(FONT_HEIGHT, character_images) if LOCALE_US else generate_plate(FONT_HEIGHT, character_images)
    if plate is None:
        return None, None, None

    M, out_of_bounds, scale = make_affine_transform(plate.shape,
                                                    output_shape,
                                                    roll_range=(-0.3, 0.3),
                                                    pitch_range=(-0.2, 0.2),
                                                    yaw_range=(-1.2, 1.2),
                                                    scale_range=(0.9, 1.0),
                                                    completely_inside=True)

    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (output_shape[1], output_shape[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)

    return out, code, True


def generate_ims(num_images, output_shape):
    """
    Generate a number of number plate images.
    :param num_images:
        Number of images to generate.
    :return:
        Iterable of number plate images.
    """

    fonts, character_images = load_fonts(FONT_DIR)
    number_background_images = len(os.listdir("background_images"))
    for i in range(num_images):
        yield generate_im(character_images[random.choice(fonts)], number_background_images, output_shape)


def generate_detect_ims(num_images, output_shape):
    fonts, character_images = load_fonts(FONT_DIR)
    number_background_images = len(os.listdir("background_images"))
    for i in range(num_images):
        yield generate_detect_im(character_images[random.choice(fonts)], number_background_images, output_shape)


def generate_image(char_ims, num_bg_images):
    bg = generate_background(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims, '6283185')

    M, out_of_bounds = make_affine_transform(
        from_shape=plate.shape,
        to_shape=bg.shape,
        min_scale=0.8,
        max_scale=0.9,
        rotation_variation=0.3,
        scale_variation=1.0,
        translation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1 - plate_mask)
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)
    return out, code, not out_of_bounds


def generate_images(num_images):
    """
    Generate a number of number plate images.

    :param num_images:
        Number of images to generate.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    # character_images = get_all_font_character_images(FONT_HEIGHT)
    fonts, character_images = load_fonts(FONT_DIR)
    number_background_images = len(os.listdir("background_images"))
    for i in range(num_images):
        yield generate_image(character_images[random.choice(fonts)], number_background_images)


if __name__ == "__main__":
    # os.mkdir("test")

    # os.mkdir("test/read")
    im_gen = generate_ims(1, READ_OUTPUT_SHAPE)
    for img_idx, (im, c, p) in enumerate(im_gen):
        if im is None:
            continue
        fname = "test/read/{:08d}_{}_{}.png".format(img_idx, c,
                                                    "1" if p else "0")
        print (fname)
        # cv2.imwrite(fname, im * 255.)

        cv2.imshow('plates', im)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    # os.mkdir("test/detect")
    im_gen = generate_detect_ims(100, DETECT_OUTPUT_SHAPE)
    for img_idx, (im, centre, scale) in enumerate(im_gen):
        if im is None:
            continue
        centre_x, centre_y = centre.flatten()
        fname = "test/detect/{:08d}_{:.3f}_{:.3f}_{:.3f}.png".format(img_idx,
                                                                     centre_x,
                                                                     centre_y,
                                                                     scale)
        print (fname)
        # cv2.imwrite(fname, im * 255.)

        size = im.shape
        rect_size = (int(size[0] * scale), int(size[1] * scale))
        pt1 = ( int(centre_x * size[1] - rect_size[1]/2), int(centre_y * size[0] - rect_size[0]/2) )
        pt2 = ( int(centre_x * size[1] + rect_size[1]/2), int(centre_y * size[0] + rect_size[0]/2) )

        cv2.rectangle(im, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow('plates', im)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

