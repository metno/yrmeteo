import os
import sys
import numpy as np

def get_code(temperature, precipitation, cloud_cover):
   assert(not np.isnan(temperature))
   assert(not np.isnan(precipitation))
   assert(not np.isnan(cloud_cover))
   if cloud_cover <= 0.13:
      C = 0
   elif cloud_cover <= 0.38:
      C = 1
   elif cloud_cover <= 0.86:
      C = 2
   else:
      C = 3

   P = get_drops(precipitation)

   if temperature < 0:
      phase = 0
   elif temperature < 1:
      phase = 1
   else:
      phase = 2

   # Cannot have light clouds with rain
   if C <= 1 and P > 0:
      C = 2

   # Cloud, precip, phase
   symbolmap = {(0,0,0) : 1, (1,0,0) : 2, (2,0,0) : 3, (3,0,0) : 4,
         (2,2,2) : 5, (2,2,1) : 7, (2,2,0) : 8,
         (3,2,2) : 9, (3,3,2) : 10, (3,2,1) : 12, (3,2,0) : 13,
         (1,1,2) : 40, (2,3,2) : 41, (2,1,1) : 42, (2,3,1) : 43,
         (2,1,0) : 44, (2,3,0) : 45, (3,1,2) : 46, (3,1,1) : 47,
         (3,3,1) : 48, (3,1,0) : 49, (3,3,0) : 50}

   if (C,P,phase) not in symbolmap:
      code = symbolmap[(C,P,0)]
   else:
      code = symbolmap[(C,P,phase)]
   return code


def get(temperature, precipitation, cloud_cover):
   size = 48
   code = get_code(temperature, precipitation, cloud_cover)

   if cloud_cover > 0.86:
      time = ""
   else:
      time = "d"
   code = "%02d%s" % (code, time)
   base = get_base()

   dir = "%s/weather-symbols/dist/png/%d" % (base, size)
   return "%s/%s.png" % (dir, code)


def get_base():
   root = __file__
   if os.path.islink(root):
      root = os.path.realpath(root)
   base = os.path.dirname(os.path.abspath(root))
   return base


def get_drops(precipitation):
   """
   Arguments:
      precip (float): Precipitation in mm
   Returns:
      int: Number of drops in the symbol
   """
   if precipitation < 0.1:
      P = 0
   elif precipitation < 0.25:
      P = 1
   elif precipitation < 0.95:
      P = 2
   else:
      P = 3
   return P
