import numpy as np
from skimage.filters import gaussian

sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])


def multivariate_gaussian(N, sigma=2):
    t = 4
    X = np.linspace(-t, t, N)
    Y = np.linspace(-t, t, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    mu = np.array([0., 0.])
    sigma = np.array([[sigma, 0], [0, sigma]])
    n = mu.shape[0]
    Sigma_det = np.linalg.det(sigma)
    Sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N


def crop_paste(img, c, N=13, sigma=2):
    Z = multivariate_gaussian(N, sigma)

    H = img.shape[1]
    W = img.shape[0]

    h = (Z.shape[0] - 1) / 2

    N = Z.shape[0]
    x1 = (c[0] - h)
    y1 = (c[1] - h)

    x2 = (c[0] + h) + 1
    y2 = (c[1] + h) + 1

    zx1 = 0
    zy1 = 0
    zx2 = N + 1
    zy2 = N + 1

    if x1 < 0:
        x1 = 0
        zx1 = 0 - (c[0] - h)

    if y1 < 0:
        y1 = 0
        zy1 = 0 - (c[1] - h)

    if x2 > W - 1:
        x2 = W - 1
        zx2 = x2 - x1 + 1
        x2 = W

    if y2 > H - 1:
        y2 = H - 1
        zy2 = y2 - y1 + 1
        y2 = H

    img[x1:x2, y1:y2] = np.maximum(Z[zx1:zx2, zy1:zy2], img[x1:x2, y1:y2])


'''
def gaussian(img, N = 13, sigma=2):
    cs = np.where(img==1)
    img = np.zeros_like(img)
    for c in zip(cs[0], cs[1]):
        crop_paste(img, c, N, sigma)
    return img
'''


def gaussian_multi_input_mp(inp):
    '''
    :param inp: Multi person ground truth heatmap input (17 ch) Each channel contains multiple joints.
    :return: out: Gaussian augmented output. Values are between 0. and 1.
    '''

    h, w, ch = inp.shape
    out = np.zeros_like(inp)
    for i in range(ch):
        layer = inp[:, :, i]
        ind = np.argwhere(layer == 1)
        b = []
        if len(ind) > 0:
            for j in ind:
                t = np.zeros((h, w))
                t[j[0], j[1]] = 1
                t = gaussian(t, sigma=sigmas[i], mode='constant')
                t = t * (1 / t.max())
                b.append(t)

            out[:, :, i] = np.maximum.reduce(b)
        else:
            out[:, :, i] = np.zeros((h, w))
    return out


def gaussian_multi_output(inp):
    '''
    :param inp: Single person ground truth heatmap input (17 ch) Each channel contains one joint.
    :return: out: Gaussian augmented output. Values are between 0. and 1.
    '''
    h, w, ch = inp.shape
    out = np.zeros_like(inp)
    for i in range(ch):
        j = np.argwhere(inp[:, :, i] == 1)
        if len(j) == 0:
            out[:, :, i] = np.zeros((h, w))
            continue
        j = j[0]
        t = np.zeros((h, w))
        t[j[0], j[1]] = 1
        t = gaussian(t, sigma=sigmas[i], mode='constant')
        out[:, :, i] = t * (1 / t.max())
    return out


def crop(img, c, N=13):
    H = img.shape[1]
    W = img.shape[0]

    h = (N - 1) / 2

    x1 = (c[0] - h)
    y1 = (c[1] - h)

    x2 = (c[0] + h) + 1
    y2 = (c[1] + h) + 1

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x2 > W - 1:
        x2 = W

    if y2 > H - 1:
        y2 = H

    return img[x1:x2, y1:y2]