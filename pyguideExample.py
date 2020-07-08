from matplotlib import pyplot as plt
import numpy as np
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
import PyGuide

# note ADU <==> count (ADU is an analog to digital unit, discretized voltages)

############# CCD Parameters #########################
# for generating fake images, and also PyGuide initialization
# these are specific to a CCD and should be changed
# when you move to your device.  They are either user
# settable parameters, static paraemters that are
# included with a stat sheet
# or measurable using lab frame image analysis.
BIAS_LEVEL = 1100
GAIN = 1
READ_NOISE = 5
######################################################

############### SIMULATED PARAMETERS #################
N_STARS = 10
SKY_LEVEL = 20 # brightness of night sky
MAX_COUNTS = 2000
######################################################


########### fake image functions ##################
# https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html
def show_image(imgData):
    plt.imshow(imgData, cmap="gray")


def read_noise(image, amount, gain=GAIN):
    """
    Generate simulated read noise.

    Parameters
    ----------

    image: numpy array
        Image whose shape the noise array should match.
    amount : float
        Amount of read noise, in electrons.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    shape = image.shape

    noise = np.random.normal(scale=amount/gain, size=shape)

    return noise


def bias(image, value, realistic=False):
    """
    Generate simulated bias image.

    Parameters
    ----------

    image: numpy array
        Image whose shape the bias array should match.
    value: float
        Bias level to add.
    realistic : bool, optional
        If ``True``, add some columns with somewhat higher bias value (a not uncommon thing)
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
    bias_im = np.zeros_like(image) + value

    # If we want a more realistic bias we need to do a little more work.
    if realistic:
        shape = image.shape
        number_of_colums = 5

        # We want a random-looking variation in the bias, but unlike the readnoise the bias should
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])

        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + col_pattern

    return bias_im


def dark_current(image, current, exposure_time, gain=GAIN, hot_pixels=False):
    """
    Simulate dark current in a CCD, optionally including hot pixels.

    Parameters
    ----------

    image : numpy array
        Image whose shape the cosmic array should match.
    current : float
        Dark current, in electrons/pixel/second, which is the way manufacturers typically
        report it.
    exposure_time : float
        Length of the simulated exposure, in seconds.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    strength : float, optional
        Pixel count in the cosmic rays.
    """

    # dark current for every pixel; we'll modify the current for some pixels if
    # the user wants hot pixels.
    base_current = current * exposure_time / gain

    # This random number generation should change on each call.
    dark_im = np.random.poisson(base_current, size=image.shape)

    if hot_pixels:
        # We'll set 0.01% of the pixels to be hot; that is probably too high but should
        # ensure they are visible.
        y_max, x_max = dark_im.shape

        n_hot = int(0.0001 * x_max * y_max)

        # Like with the bias image, we want the hot pixels to always be in the same places
        # (at least for the same image size) but also want them to appear to be randomly
        # distributed. So we set a random number seed to ensure we always get the same thing.
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)

        hot_current = 10000 * current

        dark_im[[hot_y, hot_x]] = hot_current * exposure_time / gain
    return dark_im


def sky_background(image, sky_counts, gain=GAIN):
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).

    Parameters
    ----------

    image : numpy array
        Image whose shape the cosmic array should match.
    sky_counts : float
        The target value for the number of counts (as opposed to electrons or
        photons) from the sky.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    sky_im = np.random.poisson(sky_counts * gain, size=image.shape) / gain

    return sky_im


def stars(image, number, max_counts=10000, gain=GAIN):
    """
    Add some stars to the image.
    """
    # Most of the code below is a direct copy/paste from
    # https://photutils.readthedocs.io/en/stable/_modules/photutils/datasets/make.html#make_100gaussians_image

    flux_range = [max_counts/10, max_counts] # this the range for brightness, flux or counts

    y_max, x_max = image.shape
    xmean_range = [0.1 * x_max, 0.9 * x_max] # this is where on the chip they land
    ymean_range = [0.1 * y_max, 0.9 * y_max]
    xstddev_range = [4,4] # this is a proxy for gaussian width, FWHM, or focus I think.
    ystddev_range = [4,4]
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])

    sources = make_random_gaussians_table(number, params,
                                          random_state=12345)

    star_im = make_gaussian_sources_image(image.shape, sources)

    return star_im


####### begin building a fake, realistic image ######

# a blank image
synthetic_image = np.zeros([2000, 2000])

# zero image
show_image(synthetic_image)
plt.show()
plt.close()

# noisy image
noise_im = synthetic_image + read_noise(synthetic_image, READ_NOISE)
show_image(noise_im)
plt.show()
plt.close()

# bias image
bias_only = bias(synthetic_image, BIAS_LEVEL, realistic=True)
show_image(bias_only)
plt.title('Bias alone, bad columns included', fontsize='20')
plt.show()
plt.close()

# dark image
dark_exposure = 100
dark_cur = 0.1
dark_only = dark_current(synthetic_image, dark_cur, dark_exposure, hot_pixels=True)
show_image(dark_only)
title_string = 'Dark current only, {dark_cur} $e^-$/sec/pix\n{dark_exposure} sec exposure'.format(dark_cur=dark_cur, dark_exposure=dark_exposure)
plt.title(title_string, fontsize='20')
plt.show()
plt.close()

# bias + noise
bias_noise_im = noise_im + bias_only
show_image(bias_noise_im)
plt.title('Realistic bias frame (includes read noise)', fontsize='20')
plt.show()
plt.close()

# realistic dark
dark_bias_noise_im = bias_noise_im + dark_only
show_image(dark_bias_noise_im)
plt.title('Realistic dark frame \n(with bias, read noise)', fontsize='20')
plt.show()
plt.close()

# sky background
sky_only = sky_background(synthetic_image, SKY_LEVEL)
show_image(sky_only)
plt.title('Sky background only, {} counts input'.format(SKY_LEVEL), fontsize=20)
plt.show()
plt.close()

# sky dark bias and noise
sky_dark_bias_noise_im = dark_bias_noise_im + sky_only
show_image(sky_dark_bias_noise_im)
plt.title('Sky, dark, bias and noise\n(Realistic image of clouds)', fontsize=20)
plt.show()
plt.close()

# add stars
stars_only = stars(synthetic_image, N_STARS, max_counts=MAX_COUNTS)
show_image(stars_only)
plt.title('Stars only'.format(stars_only), fontsize=20)
plt.show()
plt.close()

# stars with background
stars_with_background = sky_dark_bias_noise_im + stars_only
show_image(stars_with_background)
plt.title('Stars with noise, bias, dark, sky'.format(stars_with_background), fontsize=20)
plt.show()
plt.close()

############ PyGuide Stuff for work on fake images ###################

CCDInfo = PyGuide.CCDInfo(
    bias = BIAS_LEVEL,    # image bias, in ADU
    readNoise = READ_NOISE, # read noise, in e-
    ccdGain = GAIN,  # inverse ccd gain, in e-/ADU
)

# below, mask is an array with true false values of the same shape as the image indicating
# whether or not the routine should search for stars in a given region
# this is useful if you want to design small boxes around image in which to look for stars
# but probably isn't worth the effort
centroidData, imageStats = PyGuide.findStars(
    stars_with_background,
    mask = None,
    satMask = None,
    ccdInfo = CCDInfo
    )


print("these are the %i stars pyguide found in descending order of brightness:"%len(centroidData))
for centroid in centroidData:
    # for each star, measure its shape
    shapeData = PyGuide.starShape(
        np.asarray(stars_with_background, dtype="float32"), # had to explicitly cast for some reason
        mask = None,
        xyCtr = centroid.xyCtr,
        rad = centroid.rad
    )
    if not shapeData.isOK:
        print("starShape failed: %s" % (shapeData.msgStr,))
    else:
        print("xyCenter=[%.2f, %.2f] counts(total integrated flux from star, brightness)=%.2f star ampl(amplitude of fit gaussian, peak counts)=%.1f, fwhm(width of Gaussian, focus)=%.1f, bkgnd=%.1f, chiSq=%.2f" %\
            (centroid.xyCtr[0], centroid.xyCtr[1], centroid.counts, shapeData.ampl,shapeData.fwhm, shapeData.bkgnd, shapeData.chiSq))
print()

### highlight detections
### size of green circle scales with total counts
### bigger circles for brigher stars
plt.imshow(stars_with_background, cmap="gray", vmin=200, vmax=MAX_COUNTS) # vmin/vmax help with contrast
for centroid in centroidData:
    xyCtr = centroid.xyCtr + np.array([-0.5, -0.5]) # offset by half a pixel to match imshow with 0,0 at pixel center rather than edge
    counts = centroid.counts
    plt.scatter(xyCtr[0], xyCtr[1], s=counts/MAX_COUNTS, marker="o", edgecolors="lime", facecolors="none")
plt.show()
plt.close()


