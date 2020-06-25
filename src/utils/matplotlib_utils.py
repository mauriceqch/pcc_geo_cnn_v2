import itertools
import matplotlib.pyplot as plt


def default_rc_params(rc_params):
    rc_params['axes.labelsize'] = 20
    rc_params['xtick.labelsize'] = 20
    rc_params['ytick.labelsize'] = 20
    rc_params['legend.fontsize'] = 12
    rc_params['font.family'] = 'serif'
    rc_params['font.serif'] = ['Computer Modern Roman']
    rc_params['text.usetex'] = True
    rc_params['figure.figsize'] = 7.3, 4.2
    rc_params['legend.framealpha'] = 0.65
    return rc_params


def load_rc_params(params, rc_params):
    for k, v in params.items():
        rc_params[k] = v
    return rc_params


def linestyles_cycle():
    return itertools.cycle(('-', '--', '-.'))


def markers_cycle():
    return itertools.cycle(('s', '+', 'o', '*', 'x', 'D', 'v', 'h'))


def render_legend(labels, figsize=None, orientation='horizontal'):
    assert orientation in ('horizontal', 'vertical')
    linestyles = linestyles_cycle()
    markers = markers_cycle()
    if figsize is None:
        length_estimate = 0
        for l in labels:
            length_estimate += 0.4 + 0.166 * len(l)
        figsize = (length_estimate, 0.25)

    fig = plt.figure()
    figlegend = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    lines = []
    for i, linestyle, marker in zip(range(len(labels)), linestyles, markers):
        line = ax.plot([0], [0], linestyle=linestyle, marker=marker)
        lines.append(line[0])
    params = {}
    if orientation == 'horizontal':
        params['ncol'] = len(labels)
    figlegend.legend(lines, labels, 'center', frameon=False, **params)
    fig.canvas.draw()
    figlegend.tight_layout()
    return figlegend


def set_lims(ax, lims):
    if lims is not None:
        for lim, f, key in zip(lims,
                               (ax.set_xlim, ax.set_xlim, ax.set_ylim, ax.set_ylim),
                               ('xmin', 'xmax', 'ymin', 'ymax')):
            if lim is not None and lim != 'None':
                if isinstance(lim, str):
                    lim = float(lim)
                f(**{key: lim})