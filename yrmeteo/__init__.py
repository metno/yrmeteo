import sys
import numpy as np
import yrmeteo.driver


def main():
    yrmeteo.driver.run(sys.argv)

if __name__ == '__main__':
    main()
