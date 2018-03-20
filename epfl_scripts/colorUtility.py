###########################
# CREATE LIST OF COLORS
# returns a list of n different colors (equally separated in the hsv wheel of colors)
# usage: colors = getColors(n)
#
# input:
#   n: number of colors to generate
# output:
#   colors: list with n colors in rgb-tuple format (example [(255,0,0),(0,255,0),(0,0,255)])
###########################

import colorsys #generate colors


def getColors(n):
    colors = []
    for i in range(n):
        colors.append( tuple(255*x for x in colorsys.hsv_to_rgb(i*1.0/n, 0.5, 0.5)) )
    return colors
