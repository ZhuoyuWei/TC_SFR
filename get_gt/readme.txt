This script can be used to download ground truth data from the official sources. 

Usage:

python3 get_gt.py [output file name] [start date] [end date]

This will download all known streamflow values in the given range. When [end date] is not specified, 
the script downloads the ground truth data (10 days) for the [start date] as the target date. The
script also creates two "raw" files which are created by the services used to download the data.

It may happen that URL https://dwr.state.co.us/ is not acessible at your location and the script fails. In that
case, replace get_gt.py with get_gt_anywhere.py, which is a workaround using an intermediate server to 
download data from https://dwr.state.co.us/. Use this workaround only if necessary, it may be slower than 
the original script. 


