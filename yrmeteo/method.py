import inspect
import sys
import numpy as np
import yrmeteo.symbol
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
      precip = input.get(I, J, "precipitation_amount_acc", self.hood, self.members)
      precip = precip[1:,:] - precip[:-1,:]
      N = precip.shape[0]

      # Cloud cover
      cloud_cover = input.get(I, J, "cloud_area_fraction", self.hood, self.members)
      cloud_cover = cloud_cover[0:-1,:]

      # Don't use a neighbourhood for temperature
      temperature = input.get(I, J, "air_temperature_2m", 0, self.members) - 273.15
      temperature = temperature[0:-1,:]
      # Instead just repeat the temperature for all neighbourhood points
      temperature = np.repeat(temperature, precip.shape[1]/temperature.shape[1], axis=1)

      x_wind = y_wind = wind_gust = None
      if self.show_wind:
         x_wind = input.get(I, J, "x_wind_10m", 0, self.members)
         x_wind = x_wind[0:-1,:]
         x_wind = np.repeat(x_wind, precip.shape[1]/x_wind.shape[1], axis=1)
         y_wind = input.get(I, J, "y_wind_10m", 0, self.members)
         y_wind = y_wind[0:-1,:]
         y_wind = np.repeat(y_wind, precip.shape[1]/y_wind.shape[1], axis=1)
         if input.has_variable("x_wind_gust_10m") and input.has_variable("y_wind_gust_10m"):
            x_gust = input.get(I, J, "x_wind_gust_10m", 0, self.members)
            y_gust = input.get(I, J, "y_wind_gust_10m", 0, self.members)
            wind_gust = np.sqrt(x_gust**2 + y_gust**2)
            wind_gust = wind_gust[0:-1,:]
            wind_gust = np.repeat(wind_gust, precip.shape[1]/wind_gust.shape[1], axis=1)

      data = self.calc(temperature, precip, cloud_cover, x_wind, y_wind, wind_gust)

      return data

   def calc(self, temperature, precip, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
      """
      Turn a multi-variate ensemble into deterministic values

      Arguments:
         The following are 2D numpy arrays (time, member):
         temperature: temperature in degrees C
         precip: precipitation in mm
         cloud_cover: cloud area fraction (between 0 and 1)

      Returns:
         dict(str -> np.array): A dictionary with variable name to array of vaues
            Should contain these keys: temperature, precip, cloud_cover, precip_max
            Optionally include: x_wind, y_wind, wind_gust
      """
      raise NotImplementedError()

class Func(Simple):
   """
   Any method that uses a single function on an ensemble done separately for each forecast variable.

   Implement func() to make the class work.
   """
   def calc(self, temperature, precip, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
      T = temperature.shape[0]
      t0 = np.zeros(T, float)
      p0 = np.zeros(T, float)
      c0 = np.zeros(T, float)
      pmax0 = np.zeros(T, float)
      x0 = y0 = g0 = None
      if x_wind is not None:
         x0 = np.zeros(T, float)
      if y_wind is not None:
         y0 = np.zeros(T, float)
      if wind_gust is not None:
         g0 = np.zeros(T, float)
      for t in range(T):
         t0[t] = self.func(temperature[t, :])
         p0[t] = self.func(precip[t, :])
         c0[t] = self.func(cloud_cover[t, :])
         pmax0[t] = yrmeteo.util.nanpercentile(precip[t, :], 80)
         if x_wind is not None:
            x0[t] = self.func(x_wind[t, :])
         if y_wind is not None:
            y0[t] = self.func(y_wind[t, :])
         if wind_gust is not None:
            g0[t] = self.func(wind_gust[t, :])

      data = dict()
      data["temperature"] = t0
      data["precip"] = p0
      data["cloud_cover"] = c0
      data["precip_max"] = pmax0
      if x0 is not None:
         data["x_wind"] = x0
      if y0 is not None:
         data["y_wind"] = y0
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


class Consensus(Simple):
   """
   Compute symbol, then take the median of those members with the most common symbol
   """
   def calc(self, temperature, precip, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
      T = temperature.shape[0]
      M = temperature.shape[1]
      t0 = np.zeros(T, float)
      p0 = np.zeros(T, float)
      c0 = np.zeros(T, float)
      pmax0 = np.zeros(T, float)
      for t in range(T):
         cat = np.nan*np.zeros(M, float)
         symbols = np.nan*np.zeros(M, int)
         for m in range(M):
            if not np.isnan(precip[t, m]) and not np.isnan(cloud_cover[t, m]):
               symbols[m] = yrmeteo.symbol.get_code(10, precip[t, m], cloud_cover[t, m])
         I = np.where(np.isnan(symbols) == 0)[0]
         consensus = np.argmax(np.bincount(symbols[I].astype(int)))
         I = np.where(symbols == consensus)[0]
         t0[t] = np.nanmedian(temperature[t,I])
         p0[t] = np.nanmedian(precip[t,I])
         c0[t] = np.nanmedian(cloud_cover[t,I])
         pmax0[t] = yrmeteo.util.nanpercentile(precip[t,I], 80)
      data = dict()
      data["temperature"] = t0
      data["precip"] = p0
      data["cloud_cover"] = c0
      data["precip_max"] = pmax0

      return data


class ConsensusPrecip(Simple):
   def calc(self, temperature, precip, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
      """
      Find the most common number of drops (in symbol) and take the median of the
      members with this number of drops.
      """
      T = temperature.shape[0]
      M = temperature.shape[1]
      t0 = np.zeros(T, float)
      p0 = np.zeros(T, float)
      c0 = np.zeros(T, float)
      pmax0 = np.zeros(T, float)
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
         pmax0[t] = yrmeteo.util.nanpercentile(precip[t, I], 80)

      data = dict()
      data["temperature"] = t0
      data["precip"] = p0
      data["cloud_cover"] = c0
      data["precip_max"] = pmax0

      return data


class BestMember(Simple):
   """
   Pick the member that is most in the middle
   """
   def __init__(self, window_size=6):
      self.window_size = window_size

   def calc(self, temperature, precip, cloud_cover, x_wind=None, y_wind=None, wind_gust=None):
      T = temperature.shape[0]
      M = temperature.shape[1]
      t0 = np.zeros(T, float)
      p0 = np.zeros(T, float)
      c0 = np.zeros(T, float)
      pmax0 = np.zeros(T, float)

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
         pmax0[t] = yrmeteo.util.nanpercentile(precip[t,:], 80)

      data = dict()
      data["temperature"] = t0
      data["precip"] = p0
      data["cloud_cover"] = c0
      data["precip_max"] = pmax0

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
      precip_ens = input.get(I, J, "precipitation_amount_acc", self.hood, self.members)
      precip_ens = precip_ens[1:,:] - precip_ens[:-1,:]
      T = precip_ens.shape[0]

      precip_control = input.get(I, J, "precipitation_amount_acc", self.hood, control_member)
      precip_control = precip_control[1:,:] - precip_control[:-1,:]

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
