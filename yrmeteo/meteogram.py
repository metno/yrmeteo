import matplotlib.pylab as mpl
import numpy as np
import yrmeteo.symbol
import matplotlib.dates as mpldates
import matplotlib.colors
import matplotlib.image as mplimg
from pylab import rcParams
import copy
rcParams['figure.figsize'] = 11, 3

# blue = matplotlib.colors.to_hex("00b9f2")
blue = "#00b9f2"
blue = "#68CFE8"
red = "#FF3333"
gray = [0.7,0.7,0.7]
darkblue = "#0059b3"


class Meteogram(object):

   def __init__(self):
      self.title = None
      self.ylim = None
      self.debug = False

   def adjust_axes(self, ax=mpl.gca()):
      xlim = ax.get_xlim()
      L = xlim[1] - xlim[0]
      if L <= 5:
         ax.xaxis.set_major_locator(mpldates.DayLocator(interval=1))
         ax.xaxis.set_minor_locator(mpldates.HourLocator(byhour=range(0,24,2)))
         ax.xaxis.set_major_formatter(mpldates.DateFormatter('\n%a %d %b %Y'))
         ax.xaxis.set_minor_formatter(mpldates.DateFormatter('%H'))
         ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10, integer=True))
         ax.xaxis.grid(True, which='major', color=gray, zorder=3, linestyle='-', linewidth=2)
         ax.xaxis.grid(True, which='minor', color=gray, zorder=2, linestyle='-', lw=1)
         ax.yaxis.grid(True, which='major', color=gray, zorder=3, linestyle='-', lw=1)
      else:
         ax.xaxis.set_major_locator(mpldates.DayLocator(interval=1))
         ax.xaxis.set_minor_locator(mpldates.HourLocator(byhour=range(0,24,6)))
         ax.xaxis.set_major_formatter(mpldates.DateFormatter('%a\n%b %d'))
         ax.xaxis.set_minor_formatter(mpldates.DateFormatter(''))
         ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10, integer=True))
         ax.xaxis.grid(True, which='major', color=gray, zorder=3, linestyle='-', linewidth=2)
         ax.xaxis.grid(True, which='minor', color=gray, zorder=2, linestyle='-', lw=1)
         ax.yaxis.grid(True, which='major', color=gray, zorder=3, linestyle='-', lw=1)


   def plot(self, times, temperature, precip, cloud_cover, precip_max):
      """
      Plot temperature
      """
      ax1 = mpl.gca()
      Iwarm = np.where(temperature >= 0)[0]
      Tcold = copy.deepcopy(temperature)
      Tcold[Iwarm] = np.nan
      ax1.plot(times, temperature, '-', color=red, lw=2, zorder=3)
      ax1.plot(times, Tcold, '-', color=darkblue, lw=2, zorder=3)
      ax1.set_xlim([ax1.get_xlim()[0], np.max(times)])
      ax1.set_xlim([np.min(times), np.max(times)])

      """
      Plot symbols
      """
      xlim = ax1.get_xlim()
      # xlim = [np.min(times), xlim[0]+2] # Only allow 48 hour forecasts
      ylim = ax1.get_ylim()
      dx = xlim[1] - xlim[0]
      dy = ylim[1] - ylim[0]
      ylim = [ylim[0], ylim[1] + dy/5.0]
      dy = ylim[1] - ylim[0]
      dlt = times[1]-times[0]
      for t in range(len(times)):
         if dlt >= 2.0/24 or (times[t] * 24) % 2 == 1:
            if not np.isnan(temperature[t]) and not np.isnan(precip[t]) and not np.isnan(cloud_cover[t]):
               symbol = yrmeteo.symbol.get(temperature[t], precip[t], cloud_cover[t])
               image = mplimg.imread(symbol)
               h = 0.7/5*dy/dx*2.75*dlt*24  # Icon height
               if dlt <= 2.0/24:
                  extent = [times[t]-dlt,times[t]+dlt,temperature[t]+dy/15, temperature[t]+dy/15+h]
               else:
                  extent = [times[t]-dlt/2.0,times[t]+dlt/2.0,temperature[t]+dy/15, temperature[t]+dy/15+h/2.0]
               ax1.imshow(image, aspect="auto", extent=extent, zorder=10)
      self.adjust_axes(ax1)

      # Freezing line
      ax1.plot(xlim, [0,0], '-', lw=2, color=gray)

      """
      Plot precipitation
      """
      ax2 = ax1.twinx()
      precip[precip < 0.1] = 0
      if precip_max is not None:
         precip_max[precip_max < 0.1] = 0
         ax2.bar(times+0.1/24, precip_max, 0.8*dlt, color=blue, ec="white", hatch="/////", lw=0)
         for t in range(len(times)):
            if not np.isnan(precip_max[t]) and precip_max[t] > 0.1:
               mpl.text(times[t]+dlt/2.0, precip_max[t], "%0.1f" % precip_max[t], fontsize=6,
                     horizontalalignment="center", color="k")
      ax2.bar(times+0.1/24, precip, 0.8*dlt, color=blue, ec=blue)
      #ax2.set_ylabel("Precipitation (mm)")
      #ax2.set_xticks([])
      lim = [0, 10]
      if dlt > 2.0/24:
         # Compress y axis for precip for longer accumulation periods
         lim = [0, 25]
      ax1.set_xlim(xlim)
      ax1.set_ylim(ylim)
      ax2.set_ylim(lim)
      self.adjust_axes(ax2)
      for t in range(len(times)):
         if not np.isnan(precip[t]) and precip[t] > 0.1:
            mpl.text(times[t]+dlt/2.0, 0, "%0.1f" % precip[t], fontsize=6,
                  horizontalalignment="center", color="k")

      # Remove the last date label
      ticks = ax1.xaxis.get_major_ticks()
      for n in range(0, len(ticks)):
         tick = ticks[n]
         # Don't show the last label if it is unlikely to fit, because there aren't enough hours in
         # that day in the graph
         if n == len(ticks)-1 and (xlim[1] % 1) <= 0.42:
            tick.label1.set_visible(False)
         else:
            tick.label1.set_horizontalalignment('left')
      ax2.set_yticks([])

      labels = ["%d$^o$" % item for item in ax1.get_yticks()]
      ax1.set_yticklabels(labels)

      if self.title is not None:
         mpl.title(self.title)

      mpl.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.05, right=0.95)
      if self.ylim is not None:
         ax1.set_ylim(self.ylim)

   def show(self):
      mpl.show()

   def save(self, filename):
      mpl.savefig(filename, bbox_inches='tight')
