from someTools import *
from imageSeg import *



def stackLayerforSeg(date, path):
    wrapRaster(date, path)
    normBandsforSeg(date, path)
    cleanTMP(date, path)




def stackLayerforTrain(date, path):
    wrapRaster(date, path)
    normBandsforTraining(date, path)
    cleanTMP(date, path)

def extract(date, path, keyWord, dateList, fusionFlag, extractFlag, combineFlag, deleteFlag):
    if extractFlag:
        extractDataset(date, path, keyWord, dateList, fusionFlag)
    if combineFlag:
        conbineDataset(date, path, keyWord, dateList, fusionFlag)
    if deleteFlag:
        DeleteDatesetTMP(date, path, keyWord, dateList, fusionFlag)


dateList = np.array(['20210222','20210330', '20210426', '20210601', '20210623', '20210721', '20210825'])
date = '20210330'
path = r'C:\Users\songjunwu\Documents\files_Songjun\droneProcessing/'
dataPath = r'C:\Users\songjunwu\Documents\results\DMC/'

path = r'D:\drone/'
dataPath = r'D:\drone/'



#rasterMaskGenerator(dateList, path)   # when all the flights are done!


#normBandsforSeg_lowRes(path)
#imageSeg(date, path, numClusters=10, samplingNum=70, minPxls=200, lowResFlag=True)

#stackLayerforSeg(date, path)
#imageSeg(date, path, numClusters=30, samplingNum=70, minPxls=300, lowResFlag=False)

#stackLayerforTrain(date, path)
#generateNDVImap(path, date)


#saveSeg(date, path)
sampleChangedFlag = False
if sampleChangedFlag:
    readSegID(date, path, 'All')
    sortXYListforTrain(date, path, 'All')
    extract(date, path, 'All', dateList, fusionFlag=0, extractFlag=True, combineFlag=True, deleteFlag=False)
    extract(date, path, 'All', dateList, fusionFlag=1, extractFlag=True, combineFlag=True, deleteFlag=False)
    extract(date, path, 'All', dateList, fusionFlag=2, extractFlag=True, combineFlag=True, deleteFlag=False)
    
trainFlag = True
iterNum = 10
if trainFlag:
    TrainAndValid(date, path, fusionFlag=0, n_estimators=5000 ,iterNum=iterNum)
    TrainAndValid(date, path, fusionFlag=1, n_estimators=5000, iterNum=iterNum)
    TrainAndValid(date, path, fusionFlag=2, n_estimators=5000, iterNum=iterNum)


#sortXYListforPredict(date, path)
#extract(date, path, 'Predict', dateList, fusionFlag=0, extractFlag=True, combineFlag=True, deleteFlag=False)
#extract(date, path, 'Predict', dateList, fusionFlag=1, extractFlag=True, combineFlag=True, deleteFlag=False)
#extract(date, path, 'Predict', dateList, fusionFlag=2, extractFlag=True, combineFlag=True, deleteFlag=False)

predictFlag = False
if predictFlag:
    Predict(date, path, fusionFlag=0)
    cbFlag = False
    visiliseMaps(date, path, cbFlag, fusionFlag=0)

    Predict(date, path, fusionFlag=1)
    cbFlag = False
    visiliseMaps(date, path, cbFlag, fusionFlag=1)

    Predict(date, path, fusionFlag=2)
    cbFlag = False
    visiliseMaps(date, path, cbFlag, fusionFlag=2)

plotMapFlagFlag = False
if plotMapFlagFlag:
    cbFlag = True
    visiliseMaps(date, path, cbFlag, fusionFlag=2)

updateAllMapsFlag = False
if updateAllMapsFlag:
    updateAllMaps(dateList[1:], path)

plotPlotFlag = False
if plotPlotFlag:
    visilisePlots(dateList[1:], path, fusionFlag=0, iterNum=iterNum)
    visilisePlots(dateList[1:], path, fusionFlag=1, iterNum=iterNum)
    visilisePlots(dateList[1:], path, fusionFlag=2, iterNum=iterNum)

#stackDSMintoDataset(date, path, dataPath)


#  correction
dateList =    ['20210222','20210330', '20210426', '20210601', '20210623', '20210721', '20210825']
yoffsetList = [-0.9      ,-0.5      , 0         , -1.1      , -0.2      , -1.1      , -0.5]
xoffsetList = [-4.0      ,-2.8      , 0         , -2.0      , -2.1      , -3.9      , -4.5]
ifTest = False
#for i in [2]:
#    date = dateList[i]
#    yoffset = yoffsetList[i]   # left(negative), right(positive)
#    xoffset = xoffsetList[i]    # down(negative), up   (positive)
#    coordinateCorrection(date, path, int(-xoffset*10), int(yoffset*10), ifTest)