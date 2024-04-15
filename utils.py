from viresclient import SwarmRequest
import hapiclient
from hapiclient.util import pythonshell
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.model_utils import synth_values
from chaosmagpy.data_utils import mjd2000 # Modified Julian date

## Convert series to supervised learning
def series_to_supervised(data, indices, col_names, n_in_f=1, n_in_l=0, n_out=1, n_out_steps=1, dropnan=True):
    """
    Takes input array of data and transforms it into a supervised learning problem
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in_f, n_in_l, -1):
        cols.append(df.shift(i))
        names += [(col_names[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out, n_out_steps):
        cols.append(df.shift(-i))
        if i ==0:
            names += [(col_names[j]+'(t)') for j in range(n_vars)]
        else:
            names +=[(col_names[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = indices
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def find_events(ds, startval, interval=np.timedelta64(1,'D')):
    """Locates events based on the minimum set value for df/dt in the time interval given by the function.
        Interval has to be a np.timedelta64 object - deal with it. 
        Assumes events dataset exists somewhere."""
    events=xr.Dataset()
    pick_time = ds.where(np.abs(ds.dBdt)>startval, drop=True).Timestamp
    
    for pick in range(len(pick_time)):
        time = pick_time.values[pick]
        event = ds.sel(Timestamp=slice(time-interval,time+interval))
        events=xr.merge([events,event])
    return events


## CHAOS7 model computer
def compute_main_field(radius, colat, lon, start, end):
    ## Import model
    model=load_CHAOS_matfile('/home/s1755034/4th_project/data/CHAOS-7.4.mat')
    
    ## Get time signature in the most inconvenient way
    start_year, end_year = start[:4], end[:4]
    start_month, end_month = start[5:7], end[5:7]
    start_day, end_day = start[8:10], end[8:10]
    N=365
    
    time=np.linspace(mjd2000(start_year, start_month, start_day), mjd2000(end_year, end_month, end_day), num=N)
    
    ## Create model
    coeffs = model.synth_coeffs_tdep(time, nmax=85, deriv=0)
    B_radius, B_theta, B_phi = synth_values(coeffs, radius, colat, lon)
    
    ## Make model into dataframe
    df = pd.DataFrame(index=pd.date_range(start,end,freq='D'))
    df['B_radius']=B_radius
    df['B_phi']=B_phi
    df['B_theta']=B_theta
    CHAOS7 = df.reindex(pd.date_range(start,end,freq='1min')).interpolate()
    return CHAOS7 

## Use Ashley's fetch_observatory function (why mess with perfection)
def fetch_ground_obs(IAGA_code, start_time, end_time):
    request = SwarmRequest()
    request.set_collection(f"SW_OPER_AUX_OBSM2_:{IAGA_code}")
    request.set_products(
        measurements=["B_NEC"]
    )
    data = request.get_between(start_time, end_time)
    ds = data.as_xarray()
    ds.attrs["Latitude_GEO"] = ds["Latitude"].values[0]
    ds.attrs["Longitude_GEO"] = ds["Longitude"].values[0]
    ds.attrs["Radius_GEO"] = ds["Radius"].values[0]
    ds = ds.drop_vars(["Spacecraft", "Latitude", "Longitude", "Radius"])
    return ds

############## PROCESSING OMNI DATA ###############
## Removing bad values from OMNI
def fill2nan(hapidata,hapimeta):
    """Replace bad values (fill values given in metadata) with NaN"""
    ####FILE NO LONGER USED: Added sterp vardat=vardata.astype float, see fill2nan function. Kept for reference.#### 
    #Hapi returns metadata for parameters as
    #a list of dictionaries
    for metavar in hapimeta['parameters']:  
        varname = metavar['name']
        fillvalstr = metavar['fill']
        if fillvalstr is None:
            continue
        vardata = hapidata[varname]
        mask = vardata==float(fillvalstr)
        nbad = np.count_nonzero(mask)
        #print('{}: {} fills NaNd'.format(varname,nbad))
        vardata[mask]=np.nan
    return hapidata, hapimeta

# Get all the meta data 
def get_units_description(meta):
    units = {}
    description = {}
    for paramdict in meta["parameters"]:
        units[paramdict["name"]] = paramdict.get("units", None)
        description[paramdict["name"]] = paramdict.get("description", None)
    return units, description

# Turn into xarray
def to_xarray(df, hapimeta):
    # Here we will conveniently re-use the pandas function we just built,
    # and use the pandas API to build the xarray Dataset.
    # NB: if performance is important, it's better to build the Dataset directly
    ds = df.to_xarray()
    units, description = get_units_description(hapimeta)
    for param in ds:
        ds[param].attrs = {
            "units": units[param],
            "description": description[param]
        }
    return ds

def loadChaosModel(start_date,end_date,latitude,longitude_deg,radius=6371):
    '''Loads Chaosmagpy Model for specified time period and location at a resolution of 1 minute'''
    ##Inputs: start_date and end_date is year in format YYYY, latitude and longitude in degrees, i.e.
    ##1 degree west is 359
    
    model=load_CHAOS_matfile('data/CHAOS-7.4.mat')
    
    start_date_string= f'{start_date}-01-01'
    end_date_string=f'{end_date}-12-31'
    
    N=end_date-start_date+1
    time=np.linspace(mjd2000(start_date, 1, 1), mjd2000(end_date, 12, 31), num=N)
    theta=90.00-latitude ## Colatitude in degrees
    phi = longitude_deg ## Longitude in degrees
    
    coeffs = model.synth_coeffs_tdep(time, nmax=16, deriv=0)
    
    B_radius, B_theta, B_phi = synth_values(coeffs, radius, theta, phi)
    
    df = pd.DataFrame(index=pd.date_range(start_date_string,end_date_string,freq='1YS'))
    
    df['B_radius']=B_radius
    df['B_phi']=B_phi
    df['B_theta']=B_theta
    df['B_H']=np.sqrt(B_phi**2+B_theta**2)

    CHAOS7 = df.reindex(pd.date_range(start_date_string,end_date_string,freq='1min')).interpolate()
    return(CHAOS7)

def manipulate_obs_data(obs_data,CHAOSmodel, option =0):
    '''Inputs: obs_data, a dataframe showing the data from an observatory as downloaded using the 
    utils.fetch_ground_obs(observatory, start, end) function; CHAOSmodel, a dataframe of the
    magnetic field model in the observatory location downloaded using utils.loadChaosModel'''
    
    ###Takes the observatory data for a given station and removes estimated magnetic field. The function
    ###then subtracts the mean to get deviations around zero, before deriving a dB/dt timeseries. The
    ###uncorrected magnetic field, B_NEC, is then removed.
    ###Option: the option parameter allows you to choose the dBdt calculation style. Option 0 is default
    ###where diff (BH) is used for the calculation; this was used in Madsen et al. 2022/23. The second 
    ###option, option 1 uses sqrt(diff(BY) + diff (BX)) which allows for calculation of dBdt even
    ###at times where overall field strength is not changing but the vector direction is changing. The
    ###third option allows for download of both parameters as dBdt and dBdt1 respectively.
    
    CHAOStime=CHAOSmodel.loc[obs_data.Timestamp,:]
    
    print("Manipulating obsvervatory data")
    ## Expand B_NEC into three different variables, create horizontal field and time derivative
    obs_data = obs_data.assign(B_N=np.abs(obs_data['B_NEC'][:,0])-np.abs(CHAOStime.B_theta))
    obs_data = obs_data.assign(B_E=np.abs(obs_data['B_NEC'][:,1])-np.abs(CHAOStime.B_phi))
    obs_data = obs_data.assign(B_C=np.abs(obs_data['B_NEC'][:,2])-np.abs(CHAOStime.B_radius))
    obs_data = obs_data.assign(B_H=np.sqrt(obs_data['B_NEC'][:,0]**2+obs_data['B_NEC'][:,1]**2)-CHAOStime.B_H)
    
    ## Make sure to subtract the mean in order to get devations around zero
    obs_data = obs_data.assign(B_N=obs_data.B_N-obs_data.B_N.mean())
    obs_data = obs_data.assign(B_E=obs_data.B_E-obs_data.B_E.mean())
    obs_data = obs_data.assign(B_C=obs_data.B_C-obs_data.B_C.mean())
    obs_data = obs_data.assign(B_H=obs_data.B_H-obs_data.B_H.mean())
    
    ###Get dB/dt
    if option == 0:
        ###This means diff(BH) will be used.
        obs_data = obs_data.assign(dBdt=obs_data.B_H.diff('Timestamp')) ##OLD METHOD.
    elif option == 1:
        ###New Method.
        obs_data['dBdt1'] = (((obs_data['B_N'].diff('Timestamp') ** 2) + (obs_data['B_E'].diff('Timestamp') ** 2)) ** 0.5)
    elif option == 2:
        obs_data = obs_data.assign(dBdt=obs_data.B_H.diff('Timestamp')) 
        obs_data['dBdt1'] = (((obs_data['B_N'].diff('Timestamp') ** 2) + (obs_data['B_E'].diff('Timestamp') ** 2)) ** 0.5)
    else:
        print('ERROR! Please choose valid option between 0 and 2. refer to utility file. details in function header.')
    ###Remove uncorrected data
    obs_data = obs_data.drop(['B_NEC','NEC'])
    return(obs_data)