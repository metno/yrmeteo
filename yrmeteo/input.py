import sys
import os
import netCDF4
import numpy as np
import yrmeteo.util
import time


def distance(lat1, lon1, lat2, lon2):
    """
    Computes the great circle distance between two points using the
    haversine formula. Values can be vectors.
    """
    # Convert from degrees to radians
    pi = 3.14159265
    lon1 = lon1 * 2 * pi / 360
    lat1 = lat1 * 2 * pi / 360
    lon2 = lon2 * 2 * pi / 360
    lat2 = lat2 * 2 * pi / 360
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6.367e6 * c
    return distance


class Input(object):
    def get_times(self):
        raise NotImplementedError()

    def get(self, lat, lon, variable):
        """
        Returns:
           np.array: 2D array [time, member]
        """
        raise NotImplementedError()

    def get_lat_lon(self, i, j):
        pass


class Netcdf(object):
    def __init__(self, filename):
        self.filename = filename
        self.debug = False
        try:
            self.file = netCDF4.Dataset(self.filename, 'r')
            if "latitude" in self.file.variables:
                latvar = self.file.variables["latitude"]
                lonvar = self.file.variables["longitude"]
            else:
                latvar = self.file.variables["lat"]
                lonvar = self.file.variables["lon"]
        except:
            yrmeteo.util.error("Cannot read file '%s'" % filename)

        self.lats = latvar[:]
        self.lons = lonvar[:]
        self.isens = "ensemble_member" in self.file.dimensions

        # Store the name of the I and J dimension, so we can retrieve the data from the right
        # dimensions later on
        if len(self.lats.shape) == 1:
            self.lons,self.lats = np.meshgrid(self.lons, self.lats)
            self.Iname = latvar.dimensions[0]
            self.Jname = lonvar.dimensions[0]
        else:
            self.Iname = latvar.dimensions[0]
            self.Jname = lonvar.dimensions[1]

    def has_variable(self, name):
        if name not in self.file.variables:
            if name == "wind_speed_10m":
                return "x_wind_10m" in self.file.variables and "y_wind_10m" in self.file.variabels
            if name == "wind_speed_of_gust":
                return "x_wind_gust_10m" in self.file.variables and "y_wind_gust_10m" in self.file.variabels
        return name in self.file.variables

    def get_times(self):
        return self.file.variables["time"][:]

    def get_i_j(self, lat, lon):
        dist = distance(lat, lon, self.lats, self.lons)
        indices = np.unravel_index(dist.argmin(), dist.shape)
        X = self.lats.shape[0]
        Y = self.lats.shape[1]
        if(indices[0] > 0 and indices[0] < X-1 and indices[1] > 0 and indices[1] < Y-1):
            I = indices[0]
            J = indices[1]
        else:
            print "Station outside grid"
            I = 0
            J = 0
        return I,J

    def get(self, I, J, variable, size, members=None):
        """
        Returns a grid surrounding I,J +- size. Creates a super-ensemble by creating one
        member for each gridpoint.

        Arguments:
           I (int): I index of middle gridpoint
           J (int): J index of middle gridpoint
           size (int): Neighbourhood half size
           members (list): List of member indices. Use None for all.

        Returns:
           np.array: 2D array (time, member*I*J)
        """
        variable_use = variable
        deacc = False
        if variable not in self.file.variables:
            if variable == "precipitation_amount":
                if "precipitation_amount_acc" in self.file.variables:
                    variable_use = "precipitation_amount_acc"
                    deacc = True
                elif "precipitation_amount_consensus" in self.file.variables:
                    variable_use = "precipitation_amount_consensus"
                else:
                    yrmeteo.util.error("Cannot find precipitation variable")
            elif variable == "wind_speed_10m":
                x = self.get(I, J, "x_wind_10m", size, members)
                y = self.get(I, J, "y_wind_10m", size, members)
                return np.sqrt(x**2 + y**2)
            elif variable == "wind_gust_10m":
                x = self.get(I, J, "x_wind_gust_10m", size, members)
                y = self.get(I, J, "y_wind_gust_10m", size, members)
                return np.sqrt(x**2 + y**2)


        Irange = range(max(0, I-size), min(I+size+1, self.lats.shape[0]-1))
        Jrange = range(max(0, J-size), min(J+size+1, self.lats.shape[1]-1))

        # Determine which dimension is what
        Idim = None
        Jdim = None
        Tdim = None
        Edim = None
        dims = self.file.variables[variable_use].dimensions
        shape = self.file.variables[variable_use].shape
        for i in range(len(dims)):
            dim = dims[i]
            if dim == "time":
                Tdim = i
            elif dim == self.Iname:
                Idim = i
            elif dim == self.Jname:
                Jdim = i
            elif dim == "ensemble_member":
                Edim = i

        # Check that we have the necessary dimensions
        if Tdim is None:
            yrmeteo.util.error("Variable %s is missing a time dimension" % variable_use)
        if Idim is None:
            yrmeteo.util.error("Variable %s is missing an x dimension" % variable_use)
        if Jdim is None:
            yrmeteo.util.error("Variable %s is missing a y dimension" % variable_use)

        # Use first index for any dimension we do not recognize
        s = time.time()
        I = [[0]]*5
        I[Tdim] = range(shape[Tdim])
        I[Idim] = Irange
        I[Jdim] = Jrange
        if Edim is not None:
            if members is None:
                I[Edim] = range(shape[Edim])
            else:
                I[Edim] = members

        if len(dims) == 3:
            temp = yrmeteo.util.clean(self.file.variables[variable_use][I[0], I[1], I[2]])
            temp = np.expand_dims(temp, 3)
            temp = np.expand_dims(temp, 4)
        elif len(dims) == 4:
            temp = yrmeteo.util.clean(self.file.variables[variable_use][I[0], I[1], I[2], I[3]])
            temp = np.expand_dims(temp, 4)
        elif len(dims) == 5:
            temp = yrmeteo.util.clean(self.file.variables[variable_use][I[0], I[1], I[2], I[3], I[4]])
        else:
            yrmeteo.util.error("Variable %s does not have between 3 and 5 dimensions" % variable_use)

        # Rearrange dimensions
        EEdim = Edim
        if Edim is None:
            # Find an empty dimension that we can pretend the ensemble dimension is
            EEdim = np.setdiff1d([0,1,2,3], [Tdim,Jdim,Idim])[0]
        if self.debug:
            print temp.shape
            print [Tdim, Jdim, Idim, EEdim]
        temp = np.moveaxis(temp, [Tdim, Jdim, Idim, EEdim], [0,1,2,3])

        # Flatten array
        data = np.zeros([temp.shape[0], temp.shape[1]*temp.shape[2]*temp.shape[3]], float)
        for t in range(temp.shape[0]):
            data[t,:] = temp[t, :, :, :].flatten()

        if deacc:
            data[1:,:] = data[1:,:] - data[:-1,:]
            data[0, :] = np.nan

        return data
