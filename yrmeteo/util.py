import sys
import datetime
import matplotlib.dates
import numpy as np


def error(message):
   """ Write error message to console and abort """
   print "\033[1;31mError: " + message + "\033[0m"
   sys.exit(1)


def warning(message):
   """ Write a warning message to console """
   print "\033[1;33mWarning: " + message + "\033[0m"


def unixtime_to_datenum(time):

   """ Converts unixtime into datenum

   Arguments:
      time (int): unixtime in seconds since 1970

   Returns:
      int: datenum value
   """
   dt = datetime.datetime.utcfromtimestamp(time)
   return matplotlib.dates.date2num(dt)


def clean(data):
   """ Copy and sanitize data from a netCDF4 variable

   Arguments:
      data: A netCDF4 variable

   Returns:
      np.array: A numpy array where invalid values have been set to np.nan

   """
   data = data[:].astype(float)
   q = np.ma.filled(data, fill_value=-999)
   # Remove missing values. Convert to -999 and then back to nan to avoid
   # warning messages when doing <, >, and == comparisons with nan.
   q[np.isnan(q)] = -999
   q[(q == -999) | (q > 1e30)] = np.nan
   return q


def parse_ints(numbers, is_date=False):
   nums = parse_numbers(numbers, is_date)
   return [int(num) for num in nums]


def parse_numbers(numbers, is_date=False):
   """
   Parses numbers from an input string. Recognizes MATLAB syntax, such as:
   3              single numbers
   3,4,5          list of numbers
   3:5            number range
   3:2:12         number range with a step size of 2
   3,4:6,2:5:9,6  combinations

   Aborts if the number cannot be parsed. Expect round-off errors for values
   below about 1e-4.

   Arguments:
      numbers (str): String of numbers
      is_date (bool): True if values should be interpreted as YYYYMMDD

   Returns:
      list: parsed numbers
   """
   # Check if valid string
   if(any(char not in set('-01234567890.:,') for char in numbers)):
      error("Could not translate '" + numbers + "' into numbers")

   values = list()
   commaLists = numbers.split(',')
   for commaList in commaLists:
      colonList = commaList.split(':')
      if(len(colonList) == 1):
         values.append(float(colonList[0]))
      elif(len(colonList) <= 3):
         start = float(colonList[0])
         step = 1
         if(len(colonList) == 3):
            step = float(colonList[1])
         if step == 0:
            error("Could not parse '%s': Step cannot be 0." % (numbers))
         stepSign = step / abs(step)
         # arange does not include the end point:
         end = float(colonList[-1]) + stepSign * 0.0001
         if(is_date):
            date = min(start, end)
            curr = list()
            while date <= max(start, end):
               curr.append(date)
               date = get_date(date, step)
            values = values + list(curr)
         else:
            # Note: Values are rounded, to avoid problems with floating point
            # comparison for strings like 0.1:0.1:0.9
            values = values + list(np.round(np.arange(start, end, step), 7))
      else:
         error("Could not translate '" + numbers + "' into numbers")
      if(is_date):
         for i in range(0, len(values)):
            values[i] = int(values[i])
   return values


def nanpercentile(data, pers):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   p = np.nan
   if(len(I) > 0):
      p = np.percentile(data.flatten()[I], pers)
   return p
