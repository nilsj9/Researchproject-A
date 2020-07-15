#!/usr/bin/env python

"""
SIR modelling for COVID-19 pandemic in Germany.
"""

import parameterEstimation as pe
from lmfit import Parameters, Parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import dates
import time
import locale

###############
# DEFINITIONS #
###############

def file_reader():
    """Function to load John Hopkins german covid-19 data.

    Parameters:
        none.

    Returns:
        dataframe (dictionary): Dictionary object from Panda module containing german covid-19 data from John Hopkins university.
    """
    url_infected = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    url_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

    df_infected = pd.read_csv(url_infected)
    df_infected = df_infected[df_infected["Country/Region"] == "Germany"]
    df_infected = df_infected.drop(["Province/State", "Country/Region", "Lat", "Long"], 1)

    df_recovered = pd.read_csv(url_recovered)
    df_recovered = df_recovered[df_recovered["Country/Region"] == "Germany"]
    df_recovered = df_recovered.drop(["Province/State", "Country/Region", "Lat", "Long"], 1)

    df_death = pd.read_csv(url_death)
    df_death = df_death[df_death["Country/Region"] == "Germany"]
    df_death = df_death.drop(["Province/State", "Country/Region", "Lat", "Long"], 1)

    df_observed = pd.concat([df_infected, df_recovered, df_death])
    df_observed = df_observed.T
    df_observed.columns = ["Total_infected","Recovered","Death"]
    df_observed["Active_infected"] = df_observed.Total_infected - df_observed.Recovered - df_observed.Death
    df_observed["Daily_infected"] = df_observed.Total_infected.diff().fillna(0)
    df_observed.index.name = 'Date'
    df_observed.reset_index(inplace=True)
    df_observed['Date'] = pd.to_datetime(df_observed.Date)
    return(df_observed)

def makeDataframe(data, date):
    """Create Panda DataFrame from numpy ndarrays.

    Parameters:
        data (array): Numpy ndarray containing covid-19 data points.
        date (datetime): Datetime object containing covid-19 data timepoints.

    Returns:
        (dictionary): Dictionary object from Panda module containing covid-19 data.
    """
    df_data = pd.DataFrame(data, columns = ["Susceptible", "Exposed", "Active_infected", "Recovered", "Death"])
    df_data.columns = ["Susceptible", "Exposed", "Active_infected", "Recovered", "Death"]
    df_data["Total_infected"] = df_data.Active_infected + df_data.Recovered + df_data.Death
    df_data["Daily_infected"] = df_data.Total_infected.diff().fillna(0)
    df_data["Date"] = date
    return(df_data[["Date", "Total_infected", "Daily_infected", "Susceptible", "Exposed", "Active_infected", "Recovered", "Death"]])

def cohens_d(data1, data2):
    """Calculate cohens d effect size.

    Parameters:
        data1 (dictionary): Dictionary object from Panda module containing covid-19 data.
        data2 (dictionary): Dictionary object from Panda module containing covid-19 data.

    Returns:
        (float): Float value indicating cohens d effect size.
    """
    return((np.mean(data1) - np.mean(data2)) / (math.sqrt(((np.std(data1) ** 2) + (np.std(data2) ** 2)))))


def linePlotter(dataframe, labels, intervention=False):
    """linePlotter is a function that automatically generates a lineplot in seaborn style from panda.DataFrame.

    Parameters:
        dataframe (dictionary): Dictionary object from Panda module containing plot data specified in column "Number" corresponding to the y-axis and column "t" corresponding to x-axis.
        labels (list): List object containing label names.
        intervention (boolean): Boolean indicating intervention event.

    Returns:
        none.
    """
    marker_list = iter(["d","+","."])
    style_list = iter(["-","-.","--"])
    labels = iter(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ticklabel_format(style = 'plain')
    plt.xticks(horizontalalignment="right", rotation=45)
    plt.xlabel(next(labels))
    plt.ylabel(next(labels))

    i = 1
    for column in dataframe.iloc[:,1:]:
        if (i % 2) == 0:
            ax.plot(dataframe.Date, dataframe[column], linestyle=next(style_list), alpha=0.7, label= next(labels))
        else:
            ax.plot(dataframe.Date, dataframe[column], linestyle="", marker=next(marker_list), label = next(labels))
        i+=1
    if (intervention == True):
        ax.axvline(pd.to_datetime('2020-03-13'), color='k', linestyle='--', lw=1)

    ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(dates.WeekdayLocator(interval=1))
    leg = ax.legend(loc='upper left');
    plt.tight_layout()
    return (fig)

def barPlotter(xdata, ydata, labels):
    """Wrapper function for barplotting covid-19 data.

    Parameters:
        xdata (list): List object containing x datapoints to plot.
        ydata (list): List object containing y datapoints to plot.
        labels (list): List object containing label names.

    Returns:
        none.
    """
    fig, ax = plt.subplots()
    ax.bar(xdata, ydata)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(horizontalalignment="right", rotation=45)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(dates.WeekdayLocator(interval=2))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    return(fig)

########
# MAIN #
########

if __name__ == '__main__':

    np.random.seed(2020)
    # load data
    df_observed = file_reader().iloc[15:,:]
    df_observed.reset_index(drop=True, inplace=True)
    print("Input data:\n", df_observed, "\n")
    print("Time range: ", df_observed.Date[0], "-", df_observed.Date[len(df_observed)-1])

    # Set parameters for ODE
    # Initial Conditions
    print("\nSet initial conditions:")

    N = 83149300.0
    E0 = (4.0 * float(df_observed.Active_infected.iloc[0]) +1)
    I0 = float(df_observed.Active_infected.iloc[0])
    R0 = float(df_observed.Recovered.iloc[0])
    D0 = float(df_observed.Death.iloc[0])
    S0 = (N-(E0 + I0 + R0 + D0))

    inits = []
    inits.append(np.array([S0, E0, I0, R0, D0], dtype=np.float))

    print(f"{'  Population size N:':<35}{N:>12}",
        f"\n{'  Susceptible individuals S(0):':<35}{S0:>12}",
        f"\n{'  Exposed individuals E(0):':<35}{E0:>12}",
        f"\n{'  Active infected individuals I(0):':<35}{I0:>12}",
        f"\n{'  Recovered individuals R(0):':<35}{R0:>12}",
        f"\n{'  Death individuals D(0):':<35}{D0:>12}\n")

    # Parameter
    print("Set parameter values:")
    R_0 = 3.0
    # recovery rate
    gamma = 1.0 / 14.0
    # spreading rate
    beta = R_0 * gamma
    # latency period
    delta = 1.0 / 6.0
    # Death rate = D'(t) / I (t)
    zeta = sum(df_observed.Death.diff().fillna(0)) / sum(df_observed.Active_infected)

    print(f"\n{'  β:':<6}{beta:>12}",
          f"\n{'  γ:':<6}{gamma:>12}",
          f"\n{'  δ:':<6}{delta:>12}",
          f"\n{'  ζ:':<6}{zeta:>12}",
          f"\n{'  R_0:':<6}{R_0:>12}\n")

    # Bounds were chosen according to the RKI information platform about the covid-19 characteristics.
    params = Parameters()
    params.add("beta", value=beta, min=0, max=1)
    params.add("gamma", value=gamma, min=0, max=1)
    params.add("delta", value=delta, min=0, max=1)
    params.add("zeta", value=zeta, min=0, max=0.1)

    ##############
    # SIMULATION #
    #############

    splits = 20
    t = np.arange(0,len(df_observed))
    t_span = np.array_split(t,splits)

    print("Begin Optimization...\n")
    start = time.time()
    fitted_data, p = pe.simulation(df_observed, inits, t_span, params)
    end = time.time()
    elapsed = end - start
    print('Optimization duration %f seconds.' % elapsed)
    params["gamma"].value, params["delta"].value, params["zeta"].value = p

    simulation = []
    for i in range(200):
        params["beta"].value = np.random.normal(2.65,2.0,1) * gamma
        simulation.append(pe.odeSolver(t,inits[0],params))
    simulation = (np.mean(simulation, axis=0))

    ######################
    # Generate Plot Data #
    ######################

    # Store simulation data as Dataframe
    df_fitted = makeDataframe(fitted_data, df_observed[["Date"]])
    df_simulation = makeDataframe(simulation, df_observed[["Date"]])

    # Merge observed data with simulation data
    df_obs_app_sim = pd.concat([df_observed[["Date", "Total_infected", "Active_infected", "Daily_infected", "Recovered", "Death"]],
                                df_fitted[["Total_infected","Active_infected", "Daily_infected", "Recovered", "Death"]],
                                df_simulation[["Total_infected","Active_infected", "Daily_infected", "Recovered", "Death"]]],axis=1)
    df_obs_app_sim.columns = ["Date", "Total_infected_obs", "Active_infected_obs", "Daily_infected_obs", "Recovered_obs", "Death_obs",
                              "Total_infected_fit", "Active_infected_fit", "Daily_infected_fit", "Recovered_fit", "Death_fit",
                              "Total_infected_sim", "Active_infected_sim", "Daily_infected_sim", "Recovered_sim", "Death_sim"]

    print("\nCalculation effect of interventions")
    effect_size = cohens_d(df_obs_app_sim.Total_infected_sim, df_obs_app_sim.Total_infected_obs)
    print('Cohens d - effect size: %.3f' % effect_size)

    #########
    # PLOTS #
    #########

    # Configuration
    locale.setlocale(locale.LC_TIME,'en_US.utf8') # US dates
    sns.set(context = "paper", style="darkgrid", palette=sns.color_palette()[0:2],
            rc={"font.size":14, "axes.labelsize":14, "ytick.labelsize":12, "xtick.labelsize":12, "legend.fontsize":14, "axes.labelweight":"bold", "lines.linewidth":3, "lines.markersize":4})

    plt.rcParams["figure.figsize"] = (7,4)


    # Model fits
    fig1 = linePlotter(df_obs_app_sim[["Date", "Total_infected_obs", "Total_infected_fit", "Recovered_obs", "Recovered_fit", "Death_obs", "Death_fit"]],
                       ["Date", "Cases in Germany","Confirmed", "Confirmed fit", "Recovered", "Recovered fit", "Death", "Death fit"])
    fig1.savefig('images/fig1.pdf', dpi=300, pad_inches=0.0)
    fig2 = linePlotter(df_obs_app_sim[["Date", "Total_infected_obs", "Total_infected_fit"]],
                       ["Date", "Total confirmed cases \nin Germany", "Data", "Fit"])
    fig2.savefig('images/fig2.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig3 = linePlotter(df_obs_app_sim[["Date", "Recovered_obs", "Recovered_fit"]],
                       ["Date", "Recovered cases \nin Germany", "Data", "Fit"])
    fig3.savefig('images/fig3.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig4 = linePlotter(df_obs_app_sim[["Date", "Death_obs", "Death_fit"]],
                       ["Date", "Death cases in Germany", "Data", "Fit"])
    fig4.savefig('images/fig4.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig5 = linePlotter(df_obs_app_sim[["Date", "Active_infected_obs", "Active_infected_fit"]],
                       ["Date", "Active infected cases \nin Germany", "Data", "Fit"], True)
    fig5.savefig('images/fig5.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig6 = linePlotter(df_obs_app_sim[["Date", "Daily_infected_obs", "Daily_infected_fit"]],
                       ["Date", "Daily new infected \ncases in Germany", "Data", "Fit"], True)
    fig6.savefig('images/fig6.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)

    # Intervention Simulation
    fig7 = linePlotter(df_obs_app_sim[["Date", "Total_infected_obs","Total_infected_sim"]],
                       ["Date", "Total confirmed cases \nin Germany", "Data", "Simulation"], True)
    fig7.savefig('images/fig7.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig8 = barPlotter(df_obs_app_sim["Date"], df_obs_app_sim["Daily_infected_obs"],
                       ["Date", "Daily new infected cases \nin Germany"])
    fig8.savefig('images/bar1.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    fig9 = barPlotter(df_obs_app_sim["Date"], df_obs_app_sim["Daily_infected_sim"],
                   ["Date", "Simulated daily new \ninfected cases in Germany"])
    fig9.savefig('images/bar2.pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
