import numpy as np
import arviz as az
import geopandas as gpd
import matplotlib.pyplot as plt

import pickle
import io

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_absolute_error, r2_score


def standardize_data(data, mean=None, stddev=None):
    # Standardize features
    if mean is None and stddev is None:
        mean, stddev = np.mean(data, axis=0), np.std(data, axis=0)
    std_data = (data - mean) / stddev
    return std_data, mean, stddev


def plot_trace(trace, variables):
    """
    Plot sampling trace for given variables
    :param trace:
    :param variables:
    :return:
    """
    # Plot the trace and autocorrelation plots
    az.plot_trace(trace, var_names=variables)
    plt.show()
    

def plot_weights_bars(trace, features, var='beta'):
    # Plot some stats about the posterior of weights
    weights_traces = [np.asarray(trace['posterior'][var][:, :, i]) for i in range(len(features))]
    mean_stddev_weights = list(map(
        lambda x: (x[0], np.mean(x[1]), np.std(x[1])),
        enumerate(weights_traces)
    ))
    mean_stddev_weights = sorted(mean_stddev_weights, key=lambda x: np.abs(x[1]), reverse=True)
    labels = list(map(lambda x: features[x[0]], mean_stddev_weights))
    _, mean_weights, stddev_weights = zip(*mean_stddev_weights)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mean_weights, yerr=stddev_weights)
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()


def plot_test(pred_samples, mean, stddev, y_data, test_countries,
              title='Posterior predictive distribution (Tonnes / Ha)',
              axes=None):
    # De-standardize the predicted targets
    unnorm_pred_samples = pred_samples * stddev + mean
    if len(unnorm_pred_samples.shape) == 2:
        unnorm_pred_samples = np.expand_dims(unnorm_pred_samples, axis=0)

    figsize = 10, 8
    labels = [c for c in test_countries]
    preds = {label: unnorm_pred_samples[:, :, i] for i, label in enumerate(labels)}
    trues = {label: y_data[i] for i, label in enumerate(labels)}
    az.plot_forest(
        [preds, trues], model_names=['Prediction', 'True'], var_names=labels, transform=lambda x: 1e-4 * x,
        quartiles=True, combined=True, colors=['C0', 'C3'], figsize=figsize, ax=axes
    )
    if axes == None:
        plt.title(title)
    else:
        axes.set_title(title)


def compute_mae(pred_samples, mean, stddev, y_data):
    # De-standardize the predicted targets
    unnorm_pred_samples = pred_samples * stddev + mean

    # Get the median as predictions
    med_preds = np.median(unnorm_pred_samples.reshape(-1, len(y_data)), axis=0)

    # Compute the metrics
    mae = mean_absolute_error(y_data, med_preds)
    return mae


def get_geodata(countries):
    path = "world/World_Countries.shp"
    world = gpd.read_file(path)
    world['in_data'] = world['COUNTRY'].isin(countries)
    return world


def plot_geodata_world(countries, values=None, title=None):
    world = gpd.read_file('world/World_Countries.shp')
    world['in_data'] = world['COUNTRY'].isin(countries)
   
    if values is None:
        ax = world.plot(
            column='in_data', edgecolor='white', linewidth=0.3,
            categorical=True, cmap='tab20c_r', figsize=(20, 18)
        )
    else:
        vs = list()
        for c in world['COUNTRY']:
            if c in countries:
                i = np.argwhere(countries == c).item()
                vs.append(values[i])
            else:
                vs.append(np.nan)
        world['values'] = vs

        fig, ax = plt.subplots(figsize=(20, 18))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        ax = world.plot(
            column='values', ax=ax, cax=cax, edgecolor='white', linewidth=0.3,
            legend=True, cmap='plasma'
        )

    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    if title is not None:
        ax.set_title(title)
    plt.show()


def world_error_plot(world_df, pred_samples, mean, stddev, y_data,
                     axes=None, max_error=None):
    unnorm_pred_samples = pred_samples * stddev + mean
    # Get the median as predictions
    med_preds = np.median(unnorm_pred_samples.reshape(-1, len(y_data)), axis=0)
    error = med_preds - y_data
    world_df.loc[world_df['in_data'], 'error'] = error
    if max_error is None:
        max_error = world_df['error'].abs().max()
    ax = world_df.plot(
        ax=axes,
        column='error', legend=True, cmap='RdYlGn', vmin=-max_error, vmax=max_error,
        legend_kwds={'label': "Predictive error",
                     'orientation': "horizontal"},
        missing_kwds={
            "color": "lightgrey",
            "hatch": "///",
            "label": "Missing values",
        },
    )
    if axes is None:
        ax.figure.set_size_inches((20, 12))
