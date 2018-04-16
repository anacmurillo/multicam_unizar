"""
 Create a list of distinct colors

 normal usage:
 colors = getColors(n)

"""

import colorsys  # generate colors


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
