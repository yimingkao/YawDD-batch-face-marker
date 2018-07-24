
import os
import pandas as pd

#video_path = '../../YawDD/Mirror/Male/'
##csv_file = '../../YawDD/Mirror/Male.csv'
#csv_file = '../../YawDD/Mirror/Male_all.csv'
video_path = '../../YawDD/Mirror/Female/'
csv_file = '../../YawDD/Mirror/Female_all.csv'

data = pd.read_csv(csv_file)
for i in range(len(data)):
    #filename = video_path + data['Name'][i]+'.avi'
    filename = video_path + data['Name'][i]
    target = 'Male/'+data['Name'][i].replace('avi' ,'csv')
    if os.path.isfile(target):
        continue
    print(filename)
    os.system('..\OpenFace_2.0.3_win_x64\FeatureExtraction.exe -f "' + filename + '"')

