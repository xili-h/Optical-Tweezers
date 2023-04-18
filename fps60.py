import numpy as np

import qexpy.plotting as plt
#from scipy.optimize import curve_fit
#from scipy.stats import chisquare
import qexpy
import pandas as pd 

#Sphere appears to have diameter 0.354331 in inches, 0.009 in meters
#227 pixels per inch (ppi) for dan's macbook
#575580 Pixels per meter on computer screen
#1.9304e-10 real meters per pixel

CoversionFact = 1e-6/qexpy.Measurement(76,5)      #76 +/ 5 pixels per micron, so this is meters per pixel

def MSD_Seq(lmax,xVals):    #Returns a sequence of MSD values for each l from l = 1 to lmax
    N = len(xVals)
    MSD_seq = qexpy.MeasurementArray(np.zeros(lmax))
    #lVals = np.linspace(1,lmax,num = lmax,dtype=int)      #All values of l from 1 to lmax
    for i in range(lmax):
        x_j_l = xVals[i:]  # array with x_{l}, ..., x_{N-1}
        x_j = xVals[:N - i]  # array with x_{0}, ..., x_{N-l-1}
        SumTerms = (x_j_l - x_j) ** 2  # array with x_{l}-x_{0}, ... x_{N-1}-x_{N-l-1}
        S = np.sum(SumTerms)  # The sum
        MSD_result = S / (N - i)  
        MSD_seq[i] = MSD_result
    return MSD_seq 

def EquipStiff(xData):
    params = [289.65, 1.380649e-23, 0.01, 5e-7]
    Temp=params[0]
    kb=params[1]
    neta=params[2]
    a=params[3]

    xSampleVar = ((xData-xData.mean())**2).sum()/(len(xData)-1)    #Sample variance for x coordinates
    kx = kb*Temp/xSampleVar       #stiffness in x
    return kx

def get_time_postion(file_name):
    TrappedRaw = pd.read_csv(file_name)
    time = qexpy.MeasurementArray(TrappedRaw['time (s)'].to_numpy()[1::2], unit = "s", name = "Time")

    models = [f"model_z={z}" for z in [-200, -150, -100,-50,0,50, 100, 150, 200]]

    x_raw = {}
    y_raw = {}
    for model in models:
        x_raw[model] = TrappedRaw[f"x (pts)({model})"].to_numpy()[1::2]
        y_raw[model] = TrappedRaw[f"y (pts)({model})"].to_numpy()[1::2]
    x_raw = pd.DataFrame.from_dict(x_raw)
    y_raw = pd.DataFrame.from_dict(y_raw)


    x_avg = np.zeros(x_raw.shape[0])
    x_std = np.zeros(x_raw.shape[0])
    y_avg = np.zeros(y_raw.shape[0])
    y_std = np.zeros(y_raw.shape[0])

    for j in range(x_raw.shape[0]):
        x_avg[j] = x_raw.iloc[j].dropna().mean()
        x_std[j] = x_raw.iloc[j].dropna().std()
        y_avg[j] = y_raw.iloc[j].dropna().mean()
        y_std[j] = y_raw.iloc[j].dropna().std()


    def nan_interpolater(y):
        nans=np.isnan(y)    
        x=lambda z: z.nonzero()[0]
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return y

    x_avg = nan_interpolater(x_avg)
    x_std = nan_interpolater(x_std)
    y_avg = nan_interpolater(y_avg)
    y_std = nan_interpolater(y_std)

    x_vals = qexpy.MeasurementArray(x_avg, x_std, unit = "m", name = "x-axis Displacement")*CoversionFact
    y_vals = qexpy.MeasurementArray(y_avg, y_std, unit = "m", name = "y-axis Displacement")*CoversionFact

    return time, x_vals, y_vals

def process_data(file_name, plot_figures=False):
    time, x_vals, y_vals = get_time_postion(file_name)


    equip_stiff_x = EquipStiff(x_vals)
    equip_stiff_y = EquipStiff(y_vals)

    #if plot_figures:
    #    end = 100
    #    # Plot data on the first subplot
    #    plt.plot(time[:end], x_vals[:end], "-o")
    #    figure = plt.get_plot()
    #    figure.error_bars(False)  
    #    figure.title = 'Plot of x-axis displacement of a trapped particle'
    #    figure.show()
    #    # Plot data on the second subplot
    #    plt.plot(time[:end], y_vals[:end], "-o")
    #    figure = plt.get_plot()
    #    figure.error_bars(False)  
    #    figure.title = 'Plot of y-axis displacement of a trapped particle'
    #    figure.show()

    lmax = 50
    dt = time[3]-time[2]
    l_vals = qexpy.MeasurementArray(np.array(range(0,lmax))*dt, name="step size", unit="s")
    # Plot data on the first subplot
    MSD_x_val = qexpy.MeasurementArray(MSD_Seq(lmax,x_vals), name="$MSD_x[m^2]$")
    MSD_y_val = qexpy.MeasurementArray(MSD_Seq(lmax,y_vals), name="$MSD_y[m^2]$")

    # First define a fit model
    #theoretical model
    params = [289.65, 1.380649e-23, 0.01, 5e-7]
    Temp=params[0]
    kb=params[1]
    neta=params[2]
    a=params[3]
    def func_to_fit(tau, k): 
        y = (2*kb*Temp/k)*(1-qexpy.exp(-tau*k/(6*qexpy.pi*neta*a)))
        return y 

    #curve fitting
    #gusses: Temp = 289.65, kb = 1.380649e-23, neta = 0.01, a = 5e-7, k=1e-6
    result = qexpy.fit(
        xdata=l_vals.values[1:], 
        xerr=0, 
        ydata=MSD_x_val.values[1:],
        yerr=MSD_x_val.errors[1:], 
        model = func_to_fit, parguess=[2*kb*Temp/MSD_x_val.mean().value])
    print(result)
    #plot fitted data the the experimentla data

    if plot_figures:
        figure = plt.plot(l_vals,MSD_x_val)
        l_plot = np.linspace(0,lmax,10000)*dt
        figure.plot(l_plot, func_to_fit(l_plot, result.params[0].value),fmt="-")

        figure.error_bars()

        #figure = plt.plot(l_vals.values,fitted,fmt="-")
        figure.title = 'MSD of x-axis displacement of a trapped particle'
        figure.show()

    MSD_stiff_x = result.params[0]


    result = qexpy.fit(
        xdata=l_vals.values[1:], 
        xerr=0, 
        ydata=MSD_y_val.values[1:],
        yerr=MSD_y_val.errors[1:], 
        model = func_to_fit, parguess=[2*kb*Temp/MSD_y_val.mean().value])
    print(result)
    #plot fitted data the the experimentla data

    if plot_figures:
        figure = plt.plot(l_vals,MSD_y_val)
        l_plot = np.linspace(0,lmax,10000)*dt
        figure.plot(l_plot, func_to_fit(l_plot, result.params[0].value),fmt="-")

        figure.error_bars()

        #figure = plt.plot(l_vals.values,fitted,fmt="-")
        figure.title = 'MSD of y-axis displacement of a trapped particle'
        figure.show()

    MSD_stiff_y = result.params[0]


    return MSD_stiff_x,MSD_stiff_y,equip_stiff_x,equip_stiff_y


