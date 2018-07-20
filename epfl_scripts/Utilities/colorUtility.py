"""
 Color utilities

"""

import colorsys  # generate colors

C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)


def getColors(n):
    """
    returns a list of n different colors (equally separated in the hsv wheel of colors)
    :param n: number of colors to generate
    :return: list with n colors in rgb-tuple format (example [(255,0,0),(0,255,0),(0,0,255)])
    """
    colors = []
    for i in range(n):
        colors.append(tuple(255 * x for x in colorsys.hsv_to_rgb(i * 1.0 / n, 1, 1)))
    return colors


def blendColors(a, b, t):
    """
    Returns the blending of the colors based on a 0-1 param.
    t=0 -> return a
    t=1 -> return b
    t=0.5 -> return (a+b)/2
    etc
    :param a: first color
    :param b: second color
    :param t: 0-1 param
    :return: color
    """
    # return (math.sqrt((1 - t) * x**2 + t * y**2) for x, y in zip(a, b))
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))
