import os
import sys
from shutil import copyfile
from datetime import datetime, timedelta

import requests
import sys
from datetime import datetime, timedelta


def validate(date):
    try:
        datetime.strptime(date, '%Y-%m-%d')
        return True
    except ValueError:
        return False


tz_shifts = {"MST": 7, "MDT": 6, "CST": 6, "CDT": 5, "PST": 8, "PDT": 7, "EST": 5, "EDT": 4}
desired_times = {"00:00", "06:00", "12:00", "18:00"}

codemap = {
    "08353000": "BNDN5",
    "06468250": "ARWN8",
    "11523200": "TCCC1",
    "07301500": "CARO2",
    "06733000": "ESSC2", "BTABESCO": "ESSC2",
    "11427000": "NFDC1",
    "09209400": "LABW4",
    "06847900": "CLNK1",
    "09107000": "TRAC2",
    "06279940": "NFSW4"
}


def get_gt(argv):
    if len(argv) < 3:
        print("  Usage: python3 get_gt.py [output file name] [start date] [end date]")
        return 1
    start_date = argv[2]
    endset = True
    if len(argv) < 4:
        endset = False
    else:
        end_date = argv[3]
    file_name = argv[1]

    if not validate(start_date):
        sys.stderr.write("  \"" + start_date + "\": Incorrect date format, should be YYYY-MM-DD.")
        return 1
    if endset and not validate(end_date):
        sys.stderr.write("  \"" + end_date + "\": Incorrect date format, should be YYYY-MM-DD.")
        return 1

    start_date_obj = datetime.strptime(start_date + " 06:00", "%Y-%m-%d %H:%M")
    start_date_shift = datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M") - timedelta(hours=24)
    if endset:
        end_date_obj = datetime.strptime(end_date + " 00:00", "%Y-%m-%d %H:%M")
        end_date_shift = datetime.strptime(end_date + " 00:00", "%Y-%m-%d %H:%M") + timedelta(hours=24)
    else:
        end_date_obj = datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M") + timedelta(days=10)
        end_date_shift = datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M") + timedelta(days=10)

    link = (
            "https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&sites="
            "08353000,06468250,11523200,07301500,11427000,09209400,06847900,09107000,06279940"
            "&startDT=" + start_date_shift.strftime("%Y-%m-%d") + "T00:00%2b0000"
                                                                  "&endDT=" + end_date_shift.strftime(
        "%Y-%m-%d") + "T23:45%2b0000&parameterCd=00060&siteStatus=all"
    )
    sys.stdout.write("  Requesting the data from nwis.waterservices.usgs.gov...\n")
    with open("raw_" + file_name, "wb") as f:
        response = requests.get(link, stream=True)
        dl = 0
        for data in response.iter_content(chunk_size=65536):
            dl += len(data)
            f.write(data)
            sys.stdout.write("\r  Dowloaded so far: %s bytes" % (dl))
            sys.stdout.flush()
    sys.stdout.write("\n  Download completed!\n")

    link2 = (
            "https://dwr.state.co.us/Tools/Stations/ExportObsTsFileResult?abbrevs=BTABESCO&parameters=DISCHRG"
            "&por_start=" + start_date_shift.strftime("%Y-%m-%d") + "T00%3A00%3A00.000Z"
                                                                    "&toDate=" + end_date_shift.strftime(
        "%Y-%m-%d") + "T21%3A59%3A59.000Z"
                      "&timeStep=" + start_date_shift.strftime("%Y-%m-%d") + "T00%3A00%3A00.000Z"
                                                                             "&por_end=" + end_date_shift.strftime(
        "%Y-%m-%d") + "T21%3A59%3A59.000Z&time_step=LoggedInterval"
                      "&obs_type=best&avg_time=12%3A00%20AM&is_x_tab=true"
    )
    sys.stdout.write("  Requesting the data from dwr.state.co.us...\n")
    with open("raw2_" + file_name, "wb") as f:
        response = requests.get(link2, stream=True)
        dl = 0
        for data in response.iter_content(chunk_size=65536):
            dl += len(data)
            f.write(data)
            sys.stdout.write("\r  Dowloaded so far: %s bytes" % (dl))
            sys.stdout.flush()
    sys.stdout.write("\n  Download completed!\n")

    fw = open(file_name, "w")
    with open("raw_" + file_name, "r") as f:
        line = f.readline()
        fw.write("DateTime,LocationID,Value,Units\n")
        while line:
            if not (line.startswith("#") or line.startswith("agency") or line.startswith("5s")):
                parts = line.strip().split("\t")
                if len(parts) == 6:
                    site = parts[1]
                    time = datetime.strptime(parts[2], "%Y-%m-%d %H:%M")
                    timezone = parts[3]
                    discharge = parts[4]
                    status = parts[5]
                    if timezone in tz_shifts:
                        time += timedelta(hours=tz_shifts[timezone])
                    else:
                        sys.stderr.write("  Unexpected time zone: \"" + timezone + "\"")
                    timestr = time.strftime("%H:%M")
                    datetimestr = time.strftime("%Y-%m-%dT%H")
                    if timestr in desired_times and time >= start_date_obj and time <= end_date_obj:
                        fw.write(datetimestr + "," + codemap[site] + "," + discharge + ",CFS\n")
                else:
                    sys.stderr.write("  Unexpected line: \"" + line + "\"")
            line = f.readline()

    with open("raw2_" + file_name, "r") as f:
        line = f.readline()
        while line:
            if not (line.startswith("#") or line.startswith("abbrev")):
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    site = parts[0].strip('"')
                    time = datetime.strptime(parts[1].strip('"').strip('=').strip('"'), "%m/%d/%Y %H:%M")
                    timezone = "MDT"
                    discharge = parts[2].strip('"')
                    status = parts[4].strip('"')
                    if timezone in tz_shifts:
                        time += timedelta(hours=tz_shifts[timezone])
                    else:
                        sys.stderr.write("  Unexpected time zone: \"" + timezone + "\"")
                    timestr = time.strftime("%H:%M")
                    datetimestr = time.strftime("%Y-%m-%dT%H")
                    if timestr in desired_times and time >= start_date_obj and time <= end_date_obj:
                        fw.write(datetimestr + "," + codemap[site] + "," + discharge + ",CFS\n")
                else:
                    sys.stderr.write("  Unexpected line: \"" + line + "\"")
            line = f.readline()

    fw.close()
    return 0


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
                print("{}\t{}".format(local,value))

        print(len(local2lastest))

    return local2lastest



def download_pre_day(date):
    preday=(date - timedelta(days=1)).strftime("%Y-%m-%d")
    filename="f"+preday
    '''
    cur_path=os.getcwd()
    script_path=sys.path[0]
    os.system("cd {}".format(script_path))
    os.system("ls -l")
    os.system("python get_gt.py {} {} {}".format(filename,preday,preday))
    os.system("cd {}".format(cur_path))
    '''

    get_gt(["0",filename,preday,date.strftime("%Y-%m-%d")])
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


    with open(sys.argv[2],'w') as fout:
        fout.write("DateTime,LocationID,ForecastTime,VendorID,Value,Units\n")
        for date in target_dates:

            local2latest = download_pre_day(date)

            for site in target_sites:
                for day in range(11):
                    for time in desired_times:
                        if day == 0 and time == "00": continue
                        if day == 10 and time != "00": continue
                        value = local2latest.get(site, 0)
                        #print((date + timedelta(days=day)).strftime(
                        #    "%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime(
                        #    "%Y-%m-%dT%H") + ",TC+wzyxp_123,{},CFS".format(value)
                        fout.write((date + timedelta(days=day)).strftime(
                            "%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime(
                            "%Y-%m-%dT%H") + ",TC+wzyxp_123,{},CFS".format(value)+"\n")

    return 0


if __name__ == "__main__":
    main()