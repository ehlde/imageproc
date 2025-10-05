# Niblack

## Threshold formula:
T   = mean + k * stddev = mean + k * sqrt(variance)

Given that:
We have `sum` as a sum over an area A in the image and `sqrSum` is the sum of the square of each pixel in A.

We have:
mean      = sum / area
variance  = sqrSum/area - (sum/area)²

T  = sum/area + k * sqrt(sqrSum / area - (sum / area)²)

## Avoiding square root
TODO - see notepad.
