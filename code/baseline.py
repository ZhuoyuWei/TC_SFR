import os
import sys
from datetime import datetime, timedelta

def main(): 
    
    file = open("target_sites.csv", "r")
    target_sites = file.readlines()[1:]

    if len(sys.argv) < 2:
        print("  Usage: python3 baseline.py [target dates]")
        return 1

    print("DateTime,LocationID,ForecastTime,VendorID,Value,Units")
    target_date_str = sys.argv[1]
    target_dates_parts = target_date_str.split(',')
    target_dates = []
    for target_date in target_dates_parts:
        target_dates.append(datetime.strptime(target_date, "%Y-%m-%d"))
    desired_times = ["00", "06", "12", "18"]

    for date in target_dates:
        for site in target_sites:
            for day in range(11):
                for time in desired_times:
                    if day == 0 and time == "00": continue
                    if day == 10 and time != "00": continue
                    print((date + timedelta(days = day)).strftime("%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime("%Y-%m-%dT%H") + ",TC+wzyxp_123,0,CFS")

    return 0

if __name__ == "__main__":
    main()  