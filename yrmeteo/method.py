import inspect
import sys
import numpy as np
import yrmeteo.symbol
import copy
import re

def get_all():
    """
    Returns a dictionary of all classes where the key is the class
    name (string) and the value is the class object
    """
    temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return temp


def get(name):
    """ Returns an instance of an object with the given class name """
    methods = get_all()
    m = None
    if re.compile("member").match(name):
        num = int(name[6:])
        return Member(num)
    for method in methods:
        if(name == method[0].lower()):
            m = method[1]()
    if m is None:
        yrmeteo.util.error("Could not find method with name '%s'" % name)
    return m


class Method(object):
    def __init__(self):
        self.hood = 0
        self.members = None
        self.debug = False
        self.show_wind = False

    def get(self, input, I, J):
        """
        Get a deterministic forecast

        Arguments:
           input (yrmeteo.input): Input data set
           I (int): I coordinate
           J (int): J coordinate

        Returns:
           temperature (np.array):
           precipitation (np.array):
           cloud_cover (np.array):
           precipitation_max (np.array):
        """
        raise NotImplementedError()

    def num_points_in_hood(self, hood=None):
        if hood is None:
            hood = self.hood
        return (hood*2+1)**2


class Simple(Method):
    """
    Any method that only uses the ensemble at point I, J (plus potentially a neighbourhood) and is
    done separately on each forecast variable.

    Implement calc() to make the class work.
    """
    def get(self, input, I, J):
        # Precipitation
        precip = input.get(I, J, "precipitation_amount", self.hood, self.members)
        precip = precip[1:,:]
        N = precip.shape[0]

        # Precipitation max
        if input.has_variable("precipitation_amount_max"):
            precip_max = input.get(I, J, "precipitation_amount_max", 0, self.members)
            precip_max = precip_max[1:,:]
        elif input.has_variable("precipitation_amount_high_estimate"):
            precip_max = input.get(I, J, "precipitation_amount_high_estimate", 0, self.members)
            precip_max = precip_max[1:,:]
        else:
            yrmeteo.util.warning("Could not find precipitation_amount_max")
            precip_max = copy.deepcopy(precip)
        precip_max = np.percentile(precip_max, 80, axis=1)

        # Precipiation min
        if input.has_variable("precipitation_amount_min"):
            precip_min = input.get(I, J, "precipitation_amount_min", 0, self.members)
            precip_min = precip_min[1:,:]
        elif input.has_variable("precipitation_amount_low_estimate"):
            precip_min = input.get(I, J, "precipitation_amount_low_estimate", 0, self.members)
            precip_min = precip_min[1:,:]
        else:
            yrmeteo.util.warning("Could not find precipitation_amount_min")
            precip_min = copy.deepcopy(precip)
        precip_min = np.percentile(precip_min, 20, axis=1)

        # Probability of precipitation
        if input.has_variable("probability_of_precipitation"):
            precip_pop = input.get(I, J, "probability_of_precipitation", 0, self.members)
            precip_pop = precip_pop[1:,:]
        else:
            yrmeteo.util.warning("Could not find probability_of_precipitation")
            precip_pop = copy.deepcopy(precip)
            precip_pop = np.mean(precip_pop >= 0.1, axis=1)

        # Cloud cover
        cloud_cover = input.get(I, J, "cloud_area_fraction", self.hood, self.members)
        cloud_cover = cloud_cover[0:-1,:]

        # Don't use a neighbourhood for temperature
        temperature = input.get(I, J, "air_temperature_2m", 0, self.members) - 273.15
        temperature = temperature[0:-1,:]
        # Instead just repeat the temperature for all neighbourhood points
        temperature = np.repeat(temperature, precip.shape[1]/temperature.shape[1], axis=1)

        temperature_lower = None
        temperature_upper = None
        if input.has_variable("air_temperature_2m_lower"):
            temperature_lower = input.get(I, J, "air_temperature_2m_lower", 0, self.members) - 273.15
            temperature_lower = temperature_lower[0:-1,:]
            temperature_lower = np.repeat(temperature_lower, precip.shape[1]/temperature_lower.shape[1], axis=1)

        if input.has_variable("air_temperature_2m_upper"):
            temperature_upper = input.get(I, J, "air_temperature_2m_upper", 0, self.members) - 273.15
            temperature_upper = temperature_upper[0:-1,:]
            temperature_upper = np.repeat(temperature_upper, precip.shape[1]/temperature_upper.shape[1], axis=1)

        wind = wind_dir = wind_gust = None
        if self.show_wind:
            wind = input.get(I, J, "wind_speed_10m", 0, self.members)
            wind = wind[0:-1,:]
            wind = np.repeat(wind, precip.shape[1]/wind.shape[1], axis=1)
            wind_dir = input.get(I, J, "wind_direction_10m", 0, self.members)
            wind_dir = wind_dir[0:-1,:]
            wind_dir = np.repeat(wind_dir, precip.shape[1]/wind_dir.shape[1], axis=1)
            if input.has_variable("wind_speed_of_gust"):
                wind_gust = input.get(I, J, "wind_speed_of_gust", 0, self.members)
                wind_gust = wind_gust[0:-1,:]
                wind_gust = np.repeat(wind_gust, precip.shape[1]/wind_gust.shape[1], axis=1)

        data = self.calc(temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, wind, wind_dir, wind_gust)

        return data

    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, wind=None, wind_dir=None, wind_gust=None):
        """
        Turn a multi-variate ensemble into deterministic values

        Arguments:
           The following are 2D numpy arrays (time, member):
           temperature: temperature in degrees C
           precip: precipitation in mm
           precip_min: minimum precipitation in mm (not an ensemble)
           precip_max: maximum precipitation in mm (not an ensemble)
           precip_pop: probability of precipitation (not an ensemble)
           cloud_cover: cloud area fraction (between 0 and 1)

        Returns:
           dict(str -> np.array): A dictionary with variable name to array of vaues
              Should contain these keys: temperature, precip, cloud_cover, precip_min, precip_max, precip_pop
              Optionally include: x_wind, y_wind, wind_gust
        """
        raise NotImplementedError()

class Func(Simple):
    """
    Any method that uses a single function on an ensemble done separately for each forecast variable.

    Implement func() to make the class work.
    """
    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, wind=None, wind_dir=None, wind_gust=None):
        T = temperature.shape[0]
        t0 = np.zeros(T, float)
        t0_lower = np.zeros(T, float)
        t0_upper = np.zeros(T, float)
        p0 = np.zeros(T, float)
        c0 = np.zeros(T, float)
        pmin0 = np.zeros(T, float)
        pmax0 = np.zeros(T, float)
        pop0 = np.zeros(T, float)
        w0 = wd0 = g0 = None
        if wind is not None:
            w0 = np.zeros(T, float)
        if wind_dir is not None:
            wd0 = np.zeros(T, float)
        if wind_gust is not None:
            g0 = np.zeros(T, float)
        for t in range(T):
            t0[t] = self.func(temperature[t, :])
            if temperature_lower is not None and temperature_upper is not None:
                t0_lower[t] = self.func(temperature_lower[t, :])
                t0_upper[t] = self.func(temperature_upper[t, :])
            p0[t] = self.func(precip[t, :])
            c0[t] = self.func(cloud_cover[t, :])
            pmin0[t] = precip_min[t]
            pmax0[t] = precip_max[t]
            pop0[t] = precip_pop[t]
            if wind is not None:
                w0[t] = self.func(wind[t, :])
            if wind_dir is not None:
                wd0[t] = self.func(wind_dir[t, :])
            if wind_gust is not None:
                g0[t] = self.func(wind_gust[t, :])

        data = dict()
        data["temperature"] = t0
        if temperature_lower is not None and temperature_upper is not None:
            data["temperature_lower"] = t0_lower
            data["temperature_upper"] = t0_upper
        data["precip"] = p0
        data["cloud_cover"] = c0
        data["precip_min"] = pmin0
        data["precip_max"] = pmax0
        data["precip_pop"] = pop0
        if w0 is not None:
            data["wind"] = w0
        if wd0 is not None:
            data["wind_dir"] = wd0
        if g0 is not None:
            data["wind_gust"] = g0

        return data

    def func(self, ar):
        """
        Turn an array into a single number

        Arguments:
           ar (np.array): array of ensemble members for a single time

        Returns:
           float: A scalar value
        """
        raise NotImplementedError()

class Mean(Func):
    def func(self, ar):
        return np.nanmean(ar)


class Median(Func):
    def func(self, ar):
        return np.nanmedian(ar)

class ThomasMethod(Simple):
    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
        T = temperature.shape[0]
        M = temperature.shape[1]
        t0 = np.zeros(T, float)
        t0_lower = np.zeros(T, float)
        t0_upper = np.zeros(T, float)
        p0 = np.zeros(T, float)
        c0 = np.zeros(T, float)
        pmin0 = np.zeros(T, float)
        pmax0 = np.zeros(T, float)
        pop0 = precip_pop
        w = 6
        for t in range(w, T-w):
            I = range(t-w, t+w+1)
            N = np.nansum(precip[I, :] > 0.1, axis=0) # Number of hours with precip in the window
            N = np.nanmean(N)
            popI = np.argsort(np.argsort(pop0[I])[::-1])
            if np.round(N) > popI[w] and pop0[t] > 0:
                I = np.where(precip[t, :] > 0.1)[0]
                p0[t] = np.nanmedian(precip[t,I])

        for t in range(0, T):
            t0[t] = np.nanmedian(temperature[t, :])
            if temperature_lower is not None and temperature_upper is not None:
                t0_lower[t] = np.nanmedian(temperature_lower[t, :])
                t0_upper[t] = np.nanmedian(temperature_upper[t, :])
            #p0[t] = np.nanmedian(precip[t, :])
            c0[t] = np.nanmedian(cloud_cover[t, :])
            #pmin0[t] = precip_min[t]
            #pmax0[t] = precip_max[t]
            pop0[t] = precip_pop[t]
            pmax0[t] = precip_max[t]
            pmin0[t] = precip_min[t]
        data = dict()
        data["temperature"] = t0
        if temperature_lower is not None and temperature_upper is not None:
            data["temperature_lower"] = t0_lower
            data["temperature_upper"] = t0_upper
        data["precip"] = p0
        data["cloud_cover"] = c0
        data["precip_min"] = pmin0
        data["precip_max"] = pmax0
        data["precip_pop"] = pop0

        return data


class Consensus(Simple):
    """
    Compute symbol, then take the median of those members with the most common symbol
    """
    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
        T = temperature.shape[0]
        M = temperature.shape[1]
        t0 = np.zeros(T, float)
        t0_lower = np.zeros(T, float)
        t0_upper = np.zeros(T, float)
        p0 = np.zeros(T, float)
        c0 = np.zeros(T, float)
        pmin0 = np.zeros(T, float)
        pmax0 = np.zeros(T, float)
        pop0 = np.zeros(T, float)
        for t in range(T):
            symbols = np.nan*np.zeros(M, int)
            for m in range(M):
                if not np.isnan(precip[t, m]) and not np.isnan(cloud_cover[t, m]):
                    symbols[m] = yrmeteo.symbol.get_code(10, precip[t, m], cloud_cover[t, m])
            I = np.where(np.isnan(symbols) == 0)[0]
            consensus = np.argmax(np.bincount(symbols[I].astype(int)))
            I = np.where(symbols == consensus)[0]
            t0[t] = np.nanmedian(temperature[t, I])
            if temperature_lower is not None and temperature_upper is not None:
                t0_lower[t] = np.nanmedian(temperature_lower[t, I])
                t0_upper[t] = np.nanmedian(temperature_upper[t, I])
            p0[t] = np.nanmedian(precip[t, I])
            c0[t] = np.nanmedian(cloud_cover[t, I])
            pmin0[t] = precip_min[t]
            pmax0[t] = precip_max[t]
            pop0[t] = precip_pop[t]
        data = dict()
        data["temperature"] = t0
        if temperature_lower is not None and temperature_upper is not None:
            data["temperature_lower"] = t0_lower
            data["temperature_upper"] = t0_upper
        data["precip"] = p0
        data["cloud_cover"] = c0
        data["precip_min"] = pmin0
        data["precip_max"] = pmax0
        data["precip_pop"] = pop0

        return data


class ConsensusPrecip(Simple):
    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
        """
        Find the most common number of drops (in symbol) and take the median of the
        members with this number of drops.
        """
        T = temperature.shape[0]
        M = temperature.shape[1]
        t0 = np.zeros(T, float)
        p0 = np.zeros(T, float)
        c0 = np.zeros(T, float)
        pmin0 = np.zeros(T, float)
        pmax0 = np.zeros(T, float)
        pop0 = np.zeros(T, float)
        for t in range(T):
            # Array with number of drops for each member
            drops = np.nan*np.zeros(M, float)

            # Frequency of each number of drops
            counts = np.zeros(4)
            for m in range(0, M):
                if not np.isnan(precip[t, m]):
                    drops[m] = yrmeteo.symbol.get_drops(precip[t, m])
                    counts[int(drops[m])] += 1
            most_frequent_drops = np.argmax(counts)
            I = np.where(drops == most_frequent_drops)[0]
            t0[t] = np.nanmedian(temperature[t, I])
            p0[t] = np.nanmedian(precip[t, I])
            c0[t] = np.nanmedian(cloud_cover[t, I])
            pmin0[t] = precip_min[t]
            pmax0[t] = precip_max[t]
            pop0[t] = precip_pop[t]

        data = dict()
        data["temperature"] = t0
        data["precip"] = p0
        data["cloud_cover"] = c0
        data["precip_min"] = pmin0
        data["precip_max"] = pmax0
        data["precip_pop"] = pop0

        return data


class BestMember(Simple):
    """
    Pick the member that is most in the middle
    """
    def __init__(self, window_size=6):
        self.window_size = window_size

    def calc(self, temperature, temperature_lower, temperature_upper, precip, precip_min, precip_max, precip_pop, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
        T = temperature.shape[0]
        M = temperature.shape[1]
        t0 = np.zeros(T, float)
        p0 = np.zeros(T, float)
        c0 = np.zeros(T, float)
        pmin0 = np.zeros(T, float)
        pmax0 = np.zeros(T, float)
        pop0 = np.zeros(T, float)

        # Find optimal member
        TI = np.where(np.sum(np.isnan(precip), axis=1) == 0)[0]

        tmean = np.nanmean(temperature[TI,:], axis=1)
        pmean = np.nanmean(precip[TI,:], axis=1)
        cmean = np.nanmean(cloud_cover[TI,:], axis=1)
        te = np.mean(np.abs(temperature[TI,:] - np.reshape(np.repeat(tmean, M), [len(TI), M])), axis=0)
        pe = np.mean(np.abs(precip[TI,:] - np.reshape(np.repeat(pmean, M), [len(TI), M])), axis=0)
        ce = np.mean(np.abs(cloud_cover[TI,:] - np.reshape(np.repeat(cmean, M), [len(TI), M])), axis=0)
        total = te + pe + ce
        I = np.argmin(total)

        for t in range(T):
            t0[t] = temperature[t,I]
            p0[t] = precip[t,I]
            c0[t] = cloud_cover[t,I]
            pmin0[t] = precip_min[t]
            pmax0[t] = precip_max[t]
            pop0[t] = precip_pop[t]

        data = dict()
        data["temperature"] = t0
        data["precip"] = p0
        data["cloud_cover"] = c0
        data["precip_min"] = pmin0
        data["precip_max"] = pmax0
        data["precip_pop"] = pop0

        return data


class IvarsMethod(Simple):
    """
    Something epic
    """
    def __init__(self):
        self.min_members = 5  # Require at least this number of members before reverting to control
        self.precip_threshold = 0.1

    def get(self, input, I, J):
        data = dict()
        assert(self.members is None or len(self.members) > 1)
        control_member = [0]
        if self.members is not None:
            control_member = [self.members[0]]

        """ Precipiptation
        Use the control, except when the POP is less than 35% (Use 0 mm) or above 45% (use ensemble
        mean).

        When POP > 45% and contorl is 0, use the 60th percentile of the ensemble
        When POP < 35% and control is > 0, use 0 mm
        """
        precip_ens = input.get(I, J, "precipitation_amount", self.hood, self.members)
        precip_ens = precip_ens[1:,:]
        T = precip_ens.shape[0]

        precip_control = input.get(I, J, "precipitation_amount", self.hood, control_member)
        precip_control = precip_control[1:,:]

        pop_ens = np.mean(precip_ens >= self.precip_threshold, axis=1)
        pop_control = np.mean(precip_control >= self.precip_threshold, axis=1)

        # Base forecast base on median of control
        precip = np.percentile(precip_control, 60, axis=1)
        hood_size = (self.hood * 2 + 1) * (self.hood * 2 + 1)
        num_members = np.sum(np.isnan(precip_ens) == 0, axis=1) / hood_size
        Irm_precip = np.where((pop_ens < 0.35) & (precip > 0.1) & (num_members >= self.min_members))[0]
        Iadd_precip = np.where((pop_ens > 0.45) & (precip <= 0.1) & (num_members >= self.min_members))[0]
        print "Removing: ", Irm_precip
        print "Adding: ", Iadd_precip

        # Make adjustments
        precip[Irm_precip] = 0
        precip_ens_det = np.percentile(precip_ens, 55, axis=1)
        precip[Iadd_precip] = precip_ens_det[Iadd_precip]

        data["precip"] = precip
        # Only use precip_ens max if we don't use the control
        precip_max = np.percentile(precip_control, 80, axis=1)
        precip_max[Iadd_precip] = np.percentile(precip_ens[Iadd_precip,:], 80, axis=1)
        data["precip_max"] = precip_max

        # TODO: How to we deal with min?
        precip_min = np.percentile(precip_control, 20, axis=1)
        precip_min[Iadd_precip] = np.percentile(precip_ens[Iadd_precip,:], 20, axis=1)
        data["precip_min"] = precip_min

        data["precip_pop"] = pop_ens

        """ Cloud cover
        Use the cloud cover from the ensemble when adding precipitation
        """
        cloud_cover_ens = input.get(I, J, "cloud_area_fraction", self.hood, self.members)
        cloud_cover_ens = cloud_cover_ens[0:-1,:]
        cloud_cover_control = input.get(I, J, "cloud_area_fraction", self.hood, control_member)
        cloud_cover_control = cloud_cover_control[0:-1,:]

        cloud_cover = np.mean(cloud_cover_control, axis=1)
        # cloud_cover[Irm_precip] # Open question
        cloud_cover[Iadd_precip] = np.mean(cloud_cover_ens[Iadd_precip,:], axis=1)
        data["cloud_cover"] = cloud_cover

        """ Temperature: Use the control """
        temperature_control = input.get(I, J, "air_temperature_2m", 0, control_member) - 273.15
        data["temperature"] = temperature_control[0:-1,0]
        print "HAS?"
        if input.has_variable("air_temperature_2m_lower"):
            print "HAS"
            temperature_lower = input.get(I, J, "air_temperature_2m_lower", 0, control_member) - 273.15
            data["temperature_lower"] = temperature_lower[0:-1,0]
        if input.has_variable("air_temperature_2m_upper"):
            print "HAS"
            temperature_upper = input.get(I, J, "air_temperature_2m_upper", 0, control_member) - 273.15
            data["temperature_upper"] = temperature_upper[0:-1,0]

        if self.show_wind:
            x_wind_ens = input.get(I, J, "x_wind_ens_10m", 0, self.members)
            x_wind_ens = x_wind_ens[0:-1,:]
            y_wind_ens = input.get(I, J, "y_wind_ens_10m", 0, self.members)
            y_wind_ens = y_wind_ens[0:-1,:]
            data["x_wind"] = np.mean(x_wind_ens, axis=1)
            data["y_wind"] = np.mean(y_wind_ens, axis=1)
            if input.has_variable("x_wind_ens_gust_10m") and input.has_variable("y_wind_ens_gust_10m"):
                x_gust_ens = input.get(I, J, "x_wind_ens_gust_10m", 0, self.members)
                y_gust_ens = input.get(I, J, "y_wind_ens_gust_10m", 0, self.members)
                gust_ens = np.sqrt(x_gust_ens *2 + y_gust_ens *2)
                gust_ens = gust_ens[0:-1,:]
                data["wind_gust"] = np.mean(gust_ens, axis=1)

        return data
