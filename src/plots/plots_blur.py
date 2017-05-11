import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

np.random.seed(1979)


def demo_blur_underside():
    for _ in range(5):
        # gradient_fill(*generate_data(100), zfunc=None) # original
        gradient_fill(*generate_data(100), zfunc=zfunc)
    plt.show()


def generate_data(num):
    x = np.linspace(0, 100, num)
    y = np.random.normal(0, 1, num).cumsum()
    return x, y



def zfunc(x, y, xmin, xmax, ymin, ymax, fill_color='k', alpha=1.0):
    scale = 10
    x = (x * scale).astype(int)
    y = (y * scale).astype(int)
    # xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    # w, h = xmax - xmin, ymax - ymin
    # print w
    # print h

    h, w = 100, 1
    z = np.empty((h, w, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb

    # Build a z-alpha array which is 1 near the line and 0 at the bottom.
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    xy = (np.column_stack([x, y]))
    xy -= int(xmin), int(ymin)

    # Draw a blurred line using PIL
    draw.line(map(tuple, xy.tolist()), fill=255, width=15)
    img = img.filter(ImageFilter.GaussianBlur(radius=100))
    # Convert the PIL image to an array
    zalpha = np.asarray(img).astype(float)
    zalpha *= alpha / zalpha.max()
    # make the alphas melt to zero at the bottom
    n = zalpha.shape[0] // 4
    zalpha[:n] *= np.linspace(0, 1, n)[:, None]
    z[:, :, -1] = zalpha
    return z


def gradient_fill(x, y, y_min=None, y_max=None, fill_color=None, ax=None, zfunc=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = .5 if alpha is None else alpha

    # We assume our x-range is always the same
    xmin, xmax = x.min(), x.max()
    # But let's allow for a global y-range
    if y_min is None and y_max is None:
        ymin, ymax = y.min(), y.max()
    else:
        ymin, ymax = y_min, y_max

    if zfunc is None:
        h, w = 100, 1
        z = np.empty((h, w, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, alpha, h)[:, None]
    else:
        z = zfunc(x, y, xmin, xmax, ymin, ymax, fill_color=fill_color, alpha=alpha)

    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    ax.autoscale(True)
    return line, im


if __name__ == "__main__":
    demo_blur_underside()