import os
import pandas
import requests
import datetime
import json
import time


URL = 'https://api.data.gov.sg/v1/transport/traffic-images';
#start time
present= datetime.datetime(2021,3,1,9,0,0)


#time for which the process of taking photos repeats
#for minutes just use 'timedelta(minutes=X)'
add_time=datetime.timedelta(hours=1)

#last time
endtime=datetime.datetime(2021,3,2,0,0,0)

#to store urls and timestamps
urlsdict={}
while present<=endtime:

    #call api to collect URLs
    dateT=present.date()
    timeT=present.time()
    params={'date_time':str(dateT)+'T'+str(timeT)}
    #print(params)
    response=requests.get(url=URL,params=params)
    data=response.json()
    cams=data['items'][0]['cameras']
    targeturl=''
    for i in cams:
      if i['camera_id']=='4705':
        #print(i)
        targeturl=i['image']
        #print(targeturl)
        urlsdict[params['date_time']]=targeturl
    present=present+add_time
    if(present.hour>20):
      present = present + datetime.timedelta(days = 1)
      present= present.replace(hour=10)

print(len(urlsdict))
#batch_size==2, so make sure number of items are even 
if(len(urlsdict)%2!=0):
  urlsdict.popitem();
print(len(urlsdict))

timestamps=list(urlsdict.keys());


#download two images, detect objects and delete them
for i in range(0,len(timestamps),2):
  response=requests.get(str(urlsdict[timestamps[i]]));
  file = open("input_images_and_videos/sample_image1.png", "wb")
  file.write(response.content)
  file.close()
  response=requests.get(urlsdict[timestamps[i+1]]);
  file = open("input_images_and_videos/sample_image2.png", "wb")
  file.write(response.content)
  file.close()

  #run detection
  #call detection py file add pass timestamps for photos as arguments
  os.system("python3 single_image_object_counting.py "+timestamps[i]+" "+timestamps[i+1])

  #delete images
  os.remove("input_images_and_videos/sample_image1.png");
  os.remove("input_images_and_videos/sample_image2.png");
  time.sleep(3)