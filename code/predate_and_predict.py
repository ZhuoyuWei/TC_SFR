import os
import requests
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
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


def get_gt(argv,output_dir):
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
    sys.stderr.write("  Requesting the data from nwis.waterservices.usgs.gov...\n")
    raw_file=os.path.join(output_dir,"raw_" + file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(raw_file, "wb") as f:
        response = requests.get(link, stream=True)
        dl = 0
        for data in response.iter_content(chunk_size=65536):
            dl += len(data)
            f.write(data)
            sys.stderr.write("\r  Dowloaded so far: %s bytes" % (dl))
            sys.stderr.flush()
    sys.stderr.write("\n  Download completed!\n")

    link2 = (
            "http://13.80.70.45/tc.php?start=" + start_date_shift.strftime(
        "%Y-%m-%d") + "&end=" + end_date_shift.strftime("%Y-%m-%d")
    )
    sys.stderr.write("  Requesting the data from dwr.state.co.us...\n")
    raw_file2=os.path.join(output_dir,"raw2_" + file_name)
    with open(raw_file2, "wb") as f:
        response = requests.get(link2, stream=True)
        dl = 0
        for data in response.iter_content(chunk_size=65536):
            dl += len(data)
            f.write(data)
            sys.stderr.write("\r  Dowloaded so far: %s bytes" % (dl))
            sys.stderr.flush()
    sys.stderr.write("\n  Download completed!\n")

    final_file=os.path.join(output_dir,file_name)
    fw = open(final_file, "w")
    with open(raw_file, "r") as f:
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

    with open(raw_file2, "r") as f:
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



def download_pre_day(date,output_dir):
    preday=(date - timedelta(days=1)).strftime("%Y-%m-%d")
    filename="f"+preday
    get_gt(["0",filename,preday,date.strftime("%Y-%m-%d")],output_dir)
    return read_apires_file(os.path.join(output_dir,filename))


def download_pre_several_days(date, delta_days, output_dir):
    preday = (date - timedelta(days=delta_days)).strftime("%Y-%m-%d")
    filename = "f" + preday
    if not os.path.exists(filename):
        get_gt(["0", filename, preday, date.strftime("%Y-%m-%d")], output_dir)
    return filename





def read_ground_truth(filename):
    df=pd.read_csv(filename)
    print(df.columns)
    #print('######')
    locations=set(df["LocationID"])
    #print(locations)

    df["DateTime"]=pd.to_datetime(df["DateTime"])
    #print(df["DateTime"])
    df["Value"].astype('float')

    loc2values={}

    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        #print('$$$$$$$$$$$')
        #print(sub_df.head())

        loc2values[loc]=sub_df

    return df, loc2values

def read_ground_truth_withoutfloat(filename):
    df=pd.read_csv(filename)
    print(df.columns)
    #print('######')
    locations=set(df["LocationID"])
    #print(locations)

    df["DateTime"]=pd.to_datetime(df["DateTime"])
    #print(df["DateTime"])
    #df["Value"].astype('float')

    loc2values={}

    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        #print('$$$$$$$$$$$')
        #print(sub_df.head())

        loc2values[loc]=sub_df

    return df, loc2values
    
def convert_local2values(df):
    locations=set(df["LocationID"])
    loc2values={}
    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        loc2values[loc]=sub_df  
    return loc2values    

def produce_features(filename, max_feature, fillval, ord_encoder):
    df, local2values = read_ground_truth_withoutfloat(filename)

    df = df[df['Value'] != 'Ice']
    local2values = convert_local2values(df)


    data = []
    locations = []
    for local in local2values:
        locations.append(local)
        sub_df = local2values[local]
        sub_df = sub_df.sort_values(by='DateTime', ignore_index=True, ascending=False)
        features = sub_df['Value'].values[:max_feature].tolist()
        # print('debug features type {}'.format(features) )
        while len(features) < max_feature:
            features.append(fillval)
        data.append([local] + features)
    cat_cols = ['LocationID']
    num_cols = ["f_{}".format(i) for i in range(1, max_feature + 1)]
    header = cat_cols + num_cols

    df_res = pd.DataFrame(data, columns=header)
    df_res = ord_encoder.transform(df_res)

    return df_res, locations


def online_infer(date, lgb_loaded_model, ord_encoder, fillval, delta_days, max_feature, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = download_pre_several_days(date, delta_days, output_dir)
    df_x, locations = produce_features(os.path.join(output_dir, filename),
                                       max_feature,
                                       fillval,
                                       ord_encoder)
    num_cols = ["f_{}".format(i) for i in range(1, max_feature + 1)]
    preds = []
    while len(preds) < 40:
        dev_y_pred = lgb_loaded_model.predict(df_x)
        preds.append(dev_y_pred)

        for findex in range(len(num_cols) - 1, 0, -1):
            df_x.loc[:, num_cols[findex]] = df_x.loc[:, num_cols[findex - 1]]

        df_x.loc[:, num_cols[0]] = dev_y_pred

    local2res = {}
    for local in locations:
        local2res[local] = []
    for i in range(40):
        for j in range(len(preds[i])):
            local2res[locations[j]].append(preds[i][j])

    return local2res

def loading_models(modelfile, cefile):
    lgb_loaded_model = lgb.Booster(model_file=modelfile)

    file = open(cefile, 'rb')
    ord_encoder = pickle.load(file)
    file.close()

    return lgb_loaded_model, ord_encoder


def loading_models_locally(modeldir, locations):
    local2model = {}
    for local in locations:
        filename = os.path.join(modeldir, 'model_{}.lgb'.format(local))
        lgb_loaded_model = lgb.Booster(model_file=filename)
        local2model[local] = lgb_loaded_model

    '''
    file = open(cefile,'rb')
    ord_encoder = pickle.load(file)
    file.close()
    '''

    return local2model


def produce_features_locally(filename, max_feature, fillval):
    df, local2values = read_ground_truth(filename)
    data = []
    locations = []
    for local in local2values:
        locations.append(local)
        sub_df = local2values[local]
        sub_df = sub_df.sort_values(by='DateTime', ignore_index=True, ascending=False)
        features = sub_df['Value'].values[:max_feature].tolist()
        # print('debug features type {}'.format(features) )
        while len(features) < max_feature:
            features.append(fillval)
        data.append(features)
    # cat_cols=['LocationID']
    num_cols = ["f_{}".format(i) for i in range(1, max_feature + 1)]
    # header=cat_cols+num_cols

    df_res = pd.DataFrame(data, columns=num_cols)
    # df_res=ord_encoder.transform(df_res)

    return df_res, locations

def online_infer_locally(date, local2lgb_loaded_model, fillval, delta_days, max_feature, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = download_pre_several_days(date, delta_days, output_dir)
    df_x, locations = produce_features_locally(os.path.join(output_dir, filename),
                                               max_feature,
                                               fillval)
    num_cols = ["f_{}".format(i) for i in range(1, max_feature + 1)]
    preds = []
    while len(preds) < 40:
        dev_y_pred = []
        for i in range(len(locations)):
            lgb_loaded_model = local2lgb_loaded_model[locations[i]]
            dev_y_pred_local = lgb_loaded_model.predict(df_x.iloc[i])
            dev_y_pred.append(dev_y_pred_local)
        dev_y_pred = np.concatenate(dev_y_pred)

        preds.append(dev_y_pred)

        for findex in range(len(num_cols) - 1, 0, -1):
            df_x.loc[:, num_cols[findex]] = df_x.loc[:, num_cols[findex - 1]]

        df_x.loc[:, num_cols[0]] = dev_y_pred

    local2res = {}
    for local in locations:
        local2res[local] = []
    for i in range(40):
        for j in range(len(preds[i])):
            local2res[locations[j]].append(preds[i][j])

    return local2res


def main():
    file = open("target_sites.csv", "r")
    target_sites = file.readlines()[1:]

    if len(sys.argv) < 2:
        print("  Usage: python3 baseline.py [target dates]")
        return 1

    #print("DateTime,LocationID,ForecastTime,VendorID,Value,Units")

    '''
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
    copyfile('./get_gt.py','./tmp/get_gt.py')
    '''
    locations=[]
    for site in target_sites:
        locations.append(site.strip())
    local2lgb_loaded_model= loading_models_locally(r'/work/',locations)


    fillval = 0
    delta_days = 15
    max_feature = 50
    #output_dir = r'/vc_data/zhuwe/jupyter_sever_logs/tc_sfr/wdata/test_online_pred_v0'

    target_date_str = sys.argv[1]
    target_dates_parts = target_date_str.split(',')
    target_dates = []
    for target_date in target_dates_parts:
        target_dates.append(datetime.strptime(target_date, "%Y-%m-%d"))
    desired_times = ["00", "06", "12", "18"]

    ignore_sites=set(['NFSW4','LABW4','NFDC1','BNDN5'])


    with open(sys.argv[2],'w') as fout:

        output_dir=os.path.dirname(os.path.abspath(sys.argv[2]))

        fout.write("DateTime,LocationID,ForecastTime,VendorID,Value,Units\n")
        for date in target_dates:

            #local2latest = download_pre_day(date,output_dir)
            local2res = online_infer_locally(date, local2lgb_loaded_model, fillval, delta_days, max_feature, output_dir)

            for site in target_sites:

                values=local2res.get(site.strip(),[0]*40)
                vcount=0
                for day in range(11):
                    for time in desired_times:
                        if day == 0 and time == "00": continue
                        if day == 10 and time != "00": continue
                        #value = local2latest.get(site.strip(), 0)
                        #print((date + timedelta(days=day)).strftime(
                        #    "%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime(
                        #    "%Y-%m-%dT%H") + ",TC+wzyxp_123,{},CFS".format(value)
                        #value=values[vcount]
                        if vcount<2:
                            value=values[vcount]
                        else:
                            value=values[1]
                        #if site.strip() in ignore_sites:
                        #    value=0
                        fout.write((date + timedelta(days=day)).strftime(
                            "%Y-%m-%d") + "T" + time + "," + site.strip() + "," + date.strftime(
                            "%Y-%m-%dT%H") + ",TC+wzyxp_123,{},CFS".format(value)+"\n")
                        vcount+=1

    return 0


if __name__ == "__main__":
    main()