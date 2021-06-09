import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def test():
    print('welcome to python probabilistic ML library')


    
    
    # https://stackoverflow.com/questions/10685495/reducing-the-size-of-pdf-figure-file-in-matplotlib
    
def save_fig(fname, *args, **kwargs):
    '''Save current plot window to the figures directory.'''
    if "PYPROBML" in os.environ:
        root = os.environ["PYPROBML"]
        figdir = os.path.join(root, 'figures')
    else:
        figdir = '../figures' # default directory one above where code lives
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    fname_full = os.path.join(figdir, fname)
    print('saving image to {}'.format(fname_full))
    #plt.tight_layout()
    plt.savefig(fname_full,  dpi=300)
    
    
def savefig(fname, *args, **kwargs):
    save_fig(fname, *args, **kwargs)

    

# Source:
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor='none', edgecolor='k', plot_center='true', **kwargs):
    cov = Sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (transforms.Affine2D()
                        .rotate_deg(45)
                        .scale(scale_x, scale_y)
                        .translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)

    if plot_center:
        ax.plot(mean_x, mean_y, '.')
    return ax.add_patch(ellipse)


def convergence_test(fval, previous_fval, threshold=1e-4, warn=False):
    eps = 2e-10
    converged = 0
    delta_fval = np.abs(fval - previous_fval)
    avg_fval = (np.abs(fval) + abs(previous_fval) + eps) / 2.0
    if (delta_fval / avg_fval) < threshold:
        converged = 1

    if warn and (fval - previous_fval) < -2 * eps:
        print('convergenceTest:fvalDecrease', 'objective decreased!')
    return converged

def hinton_diagram(matrix, max_weight=None, ax=None, title=''):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix[np.isnan(matrix) == 0]).max()))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')

    for (x, y), w in np.ndenumerate(matrix):
        if w != np.nan:
            color = 'red' if w > 0 else 'green'
            size = np.sqrt(abs(w) / max_weight)
        else:
            color = 'white'
            size = 0
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.grid(linestyle='--')
    ax.set_title(title)
    ax.autoscale_view()
    ax.invert_yaxis()