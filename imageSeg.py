import rsgislib
from rsgislib.segmentation import segutils



def imageSeg(date, path, numClusters, samplingNum, minPxls, lowResFlag):
    if lowResFlag:
        inputImg = path + 'data_tmp/dataWrappedForSeg_lowRes.tif'
        outputClumps = path+'data_tmp/dataWrappedForSeg_lowRes_seg.kea'
    else:
        inputImg = path+'data_tmp/dataWrappedNorm_'+date+'.tif'
        outputClumps = path+'data_tmp/dataWrappedNorm_'+date+'_seg.kea'

    numClusters = numClusters
    minPxls = minPxls
    distThres = 125
    kmMaxIter = 3000
    sampling = samplingNum

    segutils.runShepherdSegmentation(inputImg, outputClumps, numClusters=numClusters, distThres =distThres, sampling=sampling,\
                                     kmMaxIter=kmMaxIter, minPxls=minPxls, tmpath='./segtmp')






