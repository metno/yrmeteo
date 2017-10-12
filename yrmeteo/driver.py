import sys
import argparse
import numpy as np
import yrmeteo.input
import yrmeteo.meteogram
import yrmeteo.method
import yrmeteo.util
import yrmeteo.version
import yrmeteo.symbol

def run(argv):
   parser = argparse.ArgumentParser(description="Plots an Yr-meteogram using a specified method")
   parser.add_argument('file', default=None, type=str, help="Input file (optional). Either a MEPS, AROME, or ECMWF NetCDF file.", nargs="?")
   parser.add_argument('-lat', type=float, help="Latitude in degrees")
   parser.add_argument('-lon', type=float, help="Longitude in degrees")
   parser.add_argument('-m', type=str, help="Consensus method", required=True, dest="method")
   parser.add_argument('-f', type=str, default=None, help="Output filename", dest="output_filename")
   parser.add_argument('-i', type=int, default=None, help="x-axis index")
   parser.add_argument('-j', type=int, default=None, help="y-axis index")
   parser.add_argument('-title', type=str, help="Figure title")
   parser.add_argument('-hood', default=0, type=int, help="Neighbourhood radius (in grid points)")
   parser.add_argument('-ylim', type=yrmeteo.util.parse_numbers, help="Y-axis limits for temperature (lower,upper)")
   parser.add_argument('-members', type=yrmeteo.util.parse_ints, help="Which ensemble members indices? E.g. 1:5,7,9.")
   parser.add_argument('-tz', type=int, default=0, help="Timezone (0 UTC, 1 CET)")
   parser.add_argument('--debug', help="Show debug information?", action="store_true")
   parser.add_argument('--version', action="version", version=yrmeteo.version.__version__)
   parser.add_argument('-v', help="Add extra variables (wind, gust)", dest="variables")

   if len(sys.argv) < 2:
      parser.print_help()
      sys.exit(1)

   args = parser.parse_args()

   if not args.debug:
      np.seterr(invalid="ignore")

   if args.file is None:
      yrmeteo.util.warning("No input file specified, reading latest file from thredds...")
      filename = "http://thredds.met.no/thredds/dodsC/meps25files/meps_allmembers_extracted_2_5km_latest.nc"
   else:
      filename = args.file
   input = yrmeteo.input.Netcdf(filename)
   method = yrmeteo.method.get(args.method)
   method.hood = args.hood
   method.members = args.members
   method.debug = args.debug
   method.extra_variables = args.variables
   meteo = yrmeteo.meteogram.Meteogram()
   meteo.debug = args.debug

   # meteo.0plot(input, method, args.lat, args.lon)
   if args.i is not None and args.j is not None:
      I = args.i
      J = args.j
   elif args.lat is not None and args.lon is not None:
      I,J = input.get_i_j(args.lat, args.lon)
   else:
      yrmeteo.util.error("Either -lat -lon or -i -j must be specified")

   # Timestamps
   times = input.get_times()
   times = times[:-1]
   times = np.array([yrmeteo.util.unixtime_to_datenum(time) for time in times])
   times = times+1.0*args.tz/24

   [t2m, precip, cloud_cover, precip_max, x_wind, y_wind, wind_gust] = method.get(input, I, J)

   meteo.title = args.title
   meteo.ylim = args.ylim
   meteo.plot(times, t2m, precip, cloud_cover, precip_max, x_wind, y_wind, wind_gust)
   if args.output_filename:
      meteo.save(args.output_filename)
   else:
      meteo.show()


if __name__ == '__main__':
   run(sys.argv)
