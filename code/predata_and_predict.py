import os
import sys
from shutil import copyfile
from datetime import datetime, timedelta

def read_apires_file(filename):
    local2lastest={}
    with open(filename) as f:
        header=None
        for line in f:
            if header is None:
                header = line
                continue

            ss=line.strip().split(',')
            if len(ss)==4:
                local=ss[1]
                value=0
                try:
                    value=float(ss[2])
                except:
                    continue
                local2lastest[local]=value

    return local2lastest



def download_pre_day(date):
    preday=date - timedelta(days=1).strftime("%Y-%m-%d")
    filename="./f"+preday
    cur_path=os.getcwd()
    script_path=sys.path[0]
    os.system("cd {}".format(script_path))
    os.system("ls -l")
    os.system("python get_gt.py {} {} {}".format(filename,preday,preday))
    os.system("cd {}".format(cur_path))
    return read_apires_file(filename)




def main():
    file = open("target_sites.csv", "r")
    target_sites = file.readlines()[1:]

    if len(sys.argv) < 2:
        print("  Usage: python3 baseline.py [target dates]")
        return 1

    print("DateTime,LocationID,ForecastTime,VendorID,Value,Units")

    '''
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
    copyfile('./get_gt.py','./tmp/get_gt.py')
    '''



    target_date_str = sys.argv[1]
    target_dates_parts = target_date_str.split(',')
    target_dates = []
    for target_date in target_dates_parts:
        target_dates.append(datetime.strptime(target_date, "%Y-%m-%d"))
    desired_times = ["00", "06", "12", "18"]

    for date in target_dates:

        local2latest=download_pre_day(date)

        for site in target_sites:
            for day in range(11):
                for time in desired_times:
                    if day == 0 and time == "00": continue
                    if day == 10 and time != "00": continue
                    value=local2latest.get(site,0)
                    print((date + timedelta(days=day)).strftime(
                        "%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime(
                        "%Y-%m-%dT%H") + ",TC+wzyxp_123,{},CFS".format(value))

    return 0


if __name__ == "__main__":
    main()