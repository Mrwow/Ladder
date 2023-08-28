import pandas as pd
import numpy as np
import math

def cleanAndAddcolrow(file, rownum=80):
    df = pd.read_csv(file)
    print(df.head())
    scores = []
    its = []
    indx = []
    cols = []
    rows = []
    dp_point = len(df['Score'])
    total_col = math.ceil(dp_point/rownum)
    score_map = np.empty([rownum, total_col])
    it_map = np.empty([rownum, total_col])
    indx_map = np.empty([rownum, total_col],dtype="<U10")

    for i in range(dp_point):
        # clean two or more read by selecting first read in each cell
        score = str(df["Score"][i])
        if score.find(',') != -1 :
            score = int(score.split(',')[0])
        else:
            score = int(score)
        scores.append(score)
        #
        it = str(df["IT"][i])
        if it.find(',') != -1:
            it = int(str(it).split(',')[0])
        else:
            it = int(it)
        its.append(it)

        # col_row
        col = math.ceil((i+1)/rownum)
        if (col%2) == 0:
            # even col revers order
            row = rownum - i%rownum
        else:
        # odd col order
            row = i%rownum + 1
        col_row = str(col) + "_" + str(row)

        indx.append(col_row)
        cols.append(col)
        rows.append(row)
        # map from top_right
        # score_map[col-1,row-1] = score
        # it_map[col-1,row-1] = it
        # indx_map[col-1,row-1] = str(col_row)

        #
        print(col_row)
        score_map[row-1,total_col-col] = score
        it_map[row-1,total_col-col] = it
        indx_map[row-1,total_col-col] = str(col_row)

    df['col_row'] = indx
    df['Score'] = scores
    df['IT'] = its
    df['col'] = cols
    df['row'] = rows
    outname = file.replace(".csv","clean_idx.csv")
    df.to_csv(outname, index=False,header=True)
    score_map_name = file.replace(".csv","_Map_Score.csv")
    np.savetxt(score_map_name,score_map,delimiter=',',fmt='%s')

    it_map_name = file.replace(".csv","_Map_IT.csv")
    np.savetxt(it_map_name, it_map, delimiter=',',fmt='%s')

    indx_map_name = file.replace(".csv","_Map_Index.csv")
    np.savetxt(indx_map_name, indx_map, delimiter=',',fmt='%s')





if __name__ == '__main__':
    file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/raw/2022/PCFS/nursery/winter_nursery.csv"
    cleanAndAddcolrow(file)