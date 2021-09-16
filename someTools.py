from osgeo import gdal
from gdalconst import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import shutil



def stackDSMintoDataset(date, path, dataPath):
    ds1 = gdal.Open(dataPath +'data_origin/data_' + date + '_1.tif', GA_ReadOnly)
    ds2 = gdal.Open(dataPath + 'dsm_corrected/' + date + '_dsm.tif', GA_ReadOnly)
    band = ds1.GetRasterBand(1)
    file_driver = gdal.GetDriverByName('Gtiff')
    output_dataset = file_driver.Create(path + 'data_origin/data_' + date + '.tif', band.XSize, band.YSize, 7, band.DataType)
    output_dataset.SetProjection(ds1.GetProjection())
    output_dataset.SetGeoTransform(ds1.GetGeoTransform())
    for i in range(6):
        data = ds1.GetRasterBand(i + 1).ReadAsArray()
        output_dataset.GetRasterBand(i + 1).WriteArray(data)
    data = ds2.GetRasterBand(1).ReadAsArray()
    data[data<0] = np.nan
    output_dataset.GetRasterBand(7).WriteArray(data)




def rasterMaskGenerator(dateList, path):
    for date in dateList:
        print('read mask from ', date, '...')
        wrapRaster(date, path)
        if date == '20210330':
            file_driver = gdal.GetDriverByName('Gtiff')
            dataset = gdal.Open(path + 'data_tmp/dataWrapped_20210330.tif', GA_ReadOnly)
            band = dataset.GetRasterBand(1)
            if os.path.exists(path + 'data_tmp/GPS/rasterMask.tif'):
                os.remove(path + 'data_tmp/GPS/rasterMask.tif')
            output_dataset = file_driver.Create(path + 'data_tmp/GPS/rasterMask.tif', band.XSize, band.YSize, 1, band.DataType)
            output_dataset.SetProjection(dataset.GetProjection())
            output_dataset.SetGeoTransform(dataset.GetGeoTransform())
            mask = np.full((band.YSize, band.XSize), 1)
        dataset = gdal.Open(path + 'data_tmp/dataWrapped_' + date + '.tif', GA_ReadOnly)
        data = dataset.GetRasterBand(1).ReadAsArray()
        mask[np.isnan(data)] = 0
        data = dataset.GetRasterBand(7).ReadAsArray()
        mask[np.isnan(data)] = 0
        mask[data == 0] = 0
        cleanTMP(date, path)

    output_dataset.GetRasterBand(1).WriteArray(mask)
    print('rasterMask geneerated!')




def wrapRaster(date, path):
    input_shape = path+'data_tmp/GPS/mask.shp'
    output_raster = path+'data_tmp/dataWrapped_' + date + '.tif'
    input_raster = path+'data_corrected/data_' + date + '.tif'
    ds = gdal.Warp(output_raster,
                   input_raster,
                   format='GTiff',
                   cutlineDSName=input_shape,
                   cutlineWhere="FIELD = 'whatever'",
                   cropToCutline=True,
                   dstNodata=0)
    ds = None  # close the file


def normBandsforSeg(date, path):
    dataset = gdal.Open(path+'data_tmp/dataWrapped_' + date + '.tif', GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    file_driver = gdal.GetDriverByName('Gtiff')
    if os.path.exists(path+'data_tmp/dataWrappedNorm_' + date + '.tif'):
        os.remove(path+'data_tmp/dataWrappedNorm_' + date + '.tif')

    output_dataset = file_driver.Create(path+'data_tmp/dataWrappedNorm_' + date + '.tif', band.XSize, band.YSize, 6, band.DataType)

    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    ds2 = gdal.Open(path + 'data_tmp/GPS/rasterMask.tif', GA_ReadOnly)
    rasterMask = ds2.GetRasterBand(1).ReadAsArray()

    bandList = [1,2,3,5,7]
    for i in range(6):
        if i<5:
            band = dataset.GetRasterBand(bandList[i])
            data = band.ReadAsArray()

        elif i==5:  #ndvi
            tmp1 = dataset.GetRasterBand(3).ReadAsArray()
            tmp2 = dataset.GetRasterBand(5).ReadAsArray()
            data = (tmp2 - tmp1) / (tmp2 + tmp1)

        data[data>np.nanpercentile(data,99)] = np.nanpercentile(data,99)
        data[data<np.nanpercentile(data,1)] = np.nanpercentile(data,1)
        print(i, np.nanmax(data), np.nanmin(data), np.nanpercentile(data,99), np.nanpercentile(data,1))
        data = (data-np.nanmin(data))*255/(np.nanmax(data)-np.nanmin(data))
        data[rasterMask==0] = np.nan
        output_dataset.GetRasterBand(i + 1).WriteArray(data)

def normBandsforTraining(date, path):
    dataset = gdal.Open(path+'data_tmp/dataWrapped_' + date + '.tif', GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    file_driver = gdal.GetDriverByName('Gtiff')
    if os.path.exists(path+'data_tmp/dataWrappedForTrain_' + date + '.tif'):
        os.remove(path+'data_tmp/dataWrappedForTrain_' + date + '.tif')
    output_dataset = file_driver.Create(path+'data_tmp/dataWrappedForTrain_' + date + '.tif', band.XSize, band.YSize, 7, band.DataType)
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    ds2 = gdal.Open(path + 'data_tmp/GPS/rasterMask.tif', GA_ReadOnly)
    rasterMask = ds2.GetRasterBand(1).ReadAsArray()

    for i in range(7):
        band = dataset.GetRasterBand(i+1)
        data = band.ReadAsArray()
        if i==5:
            data[data>np.nanpercentile(data,99)] = np.nanpercentile(data,99)
            data[data<np.nanpercentile(data,2)] = np.nanpercentile(data,2)
        else:
            data[data>np.nanpercentile(data,99)] = np.nanpercentile(data,99)
            data[data<np.nanpercentile(data,1)] = np.nanpercentile(data,1)

        if i==5:
            data = data/100-273.15
        data[rasterMask==0] = np.nan
        print(i, np.nanmax(data), np.nanmin(data), np.nanpercentile(data,99), np.nanpercentile(data,1))
        output_dataset.GetRasterBand(i + 1).WriteArray(data)


def generateNDVImap(path, date):
    dataset = gdal.Open(path+'data_tmp/dataWrappedForTrain_' + date + '.tif', GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    file_driver = gdal.GetDriverByName('Gtiff')
    if os.path.exists(path+'data_tmp/dataNDVI_' + date + '.tif'):
        os.remove(path+'data_tmp/dataNDVI_' + date + '.tif')
    output_dataset = file_driver.Create(path+'data_tmp/dataNDVI_' + date + '.tif', band.XSize, band.YSize, 1, band.DataType)
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    ds2 = gdal.Open(path + 'data_tmp/GPS/rasterMask.tif', GA_ReadOnly)
    rasterMask = ds2.GetRasterBand(1).ReadAsArray()
    data = (dataset.GetRasterBand(5).ReadAsArray()-dataset.GetRasterBand(3).ReadAsArray())/(dataset.GetRasterBand(5).ReadAsArray()+dataset.GetRasterBand(3).ReadAsArray())
    data[rasterMask == 0] = np.nan
    output_dataset.GetRasterBand(1).WriteArray(data)


def normBandsforSeg_lowRes(path):
    dateList = np.array(['20210330', '20210426', '20210601', '20210721', '20210825'])
    for i in range(len(dateList)):
        if i == 0:
            ds = gdal.Open(path + 'data_tmp/dataWrappedForTrain_20210330.tif', GA_ReadOnly)
            band = ds.GetRasterBand(1)
            file_driver = gdal.GetDriverByName('Gtiff')
            if os.path.exists(path + 'data_tmp/dataWrappedForSeg_lowRes.tif'):
                os.remove(path + 'data_tmp/dataWrappedForSeg_lowRes.tif')
            output_dataset = file_driver.Create(path + 'data_tmp/dataWrappedForSeg_lowRes.tif', band.XSize, band.YSize, len(dateList)+1,
                                                band.DataType)
            output_dataset.SetProjection(ds.GetProjection())
            output_dataset.SetGeoTransform(ds.GetGeoTransform())
            ds2 = gdal.Open(path + 'data_tmp/GPS/rasterMask.tif', GA_ReadOnly)
            rasterMask = ds2.GetRasterBand(1).ReadAsArray()
        ds = gdal.Open(path + 'data_tmp/dataNDVI_'+dateList[i]+'.tif', GA_ReadOnly)
        data = ds.GetRasterBand(1).ReadAsArray()
        data[rasterMask == 0] = np.nan
        output_dataset.GetRasterBand(i+1).WriteArray(data)
    ds = gdal.Open(path + 'data_tmp/dataWrappedForTrain_20210330.tif', GA_ReadOnly)
    data = ds.GetRasterBand(7).ReadAsArray()
    data[rasterMask == 0] = np.nan
    output_dataset.GetRasterBand(len(dateList)+1).WriteArray(data)

def saveSeg(date, path):

    file_driver = gdal.GetDriverByName('Gtiff')
    dataset1 = gdal.Open(path+'data_tmp/dataWrappedNorm_' + date + '.tif', GA_ReadOnly)
    band1 = dataset1.GetRasterBand(1)
    dataset2= gdal.Open(path+'data_tmp/'+date+'_seg1.tif', GA_ReadOnly)
    data2 = dataset2.GetRasterBand(1).ReadAsArray()
    output_dataset = file_driver.Create(path+'data_tmp/'+date+'_seg.tif', band1.XSize, band1.YSize, 1, band1.DataType)
    output_dataset.SetProjection(dataset1.GetProjection())
    output_dataset.SetGeoTransform(dataset1.GetGeoTransform())
    output_dataset.GetRasterBand(1).WriteArray(data2)


def readSegID(date, path, keyWord):

    ds = gdal.Open(path + 'data_tmp/'+date+'_seg.tif',GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray()
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    prosrs = ds.GetSpatialRef()
    prosrs.ImportFromWkt(ds.GetProjection())
    #geosrs = prosrs.CloneGeogCS()
    #ct = osr.CoordinateTransformation(geosrs, prosrs)

    corrdsList = np.loadtxt(path + 'data_tmp/'+keyWord+'_'+date+'.txt', delimiter=',', skiprows=1)

    xlist = corrdsList[:, 2]
    ylist = corrdsList[:, 3]

    outputList = np.array([])

    for i in range(len(ylist)):
        #coords = ct.TransformPoint(xlist[i], ylist[i])
        #xOffset = int((coords[0] - xOrigin) / pixelWidth)
        #yOffset = int((coords[1] - yOrigin) / pixelHeight)
        xOffset = int((xlist[i] - xOrigin) / pixelWidth)
        yOffset = int((ylist[i] - yOrigin) / pixelHeight)
        outputList = np.append(outputList, data[yOffset, xOffset])
    np.savetxt(path + 'data_tmp/' + keyWord+'_'+date + '_seg.txt', outputList)


def cleanTMP(date, path):
    os.remove(path+'data_tmp/dataWrapped_' + date + '.tif')

def heatmap(data):
    plt.imshow(data, cmap='viridis')
    plt.show()


def coordinateCorrection(date,path, xoffset, yoffset, ifTest):
    ds = gdal.Open(path+'data_origin/data_' + date + '.tif', GA_ReadOnly)
    band = ds.GetRasterBand(1)
    transforms = ds.GetGeoTransform()
    file_driver = gdal.GetDriverByName('Gtiff')
    if ifTest:
        output_dataset = file_driver.Create(path + 'test_' + date + '.tif', band.XSize, band.YSize, 3,
                                            band.DataType)
    else:
        output_dataset = file_driver.Create(path + 'data_corrected/data_' + date + '.tif', band.XSize, band.YSize, 7,
                                            band.DataType)
    output_dataset.SetProjection(ds.GetProjection())
    output_dataset.SetGeoTransform(transforms)
    if ifTest:
        xx = 3
    else:
        xx = 7
    for i in range(xx):
        data1 = ds.GetRasterBand(i+1).ReadAsArray()
        data2 = np.full((band.YSize,band.XSize), -9999.0)
        if xoffset > 0 and yoffset > 0:
            data2[:xoffset, :] = np.nan
            data2[:, :yoffset] = np.nan
            data2[xoffset:, yoffset:] = data1[:-xoffset, :-yoffset]
        elif xoffset < 0 and yoffset < 0:
            data2[xoffset:, :] = np.nan
            data2[:, yoffset:] = np.nan
            data2[:xoffset, :yoffset] = data1[-xoffset:, -yoffset:]
        elif xoffset > 0 and yoffset < 0:
            data2[:xoffset, :] = np.nan
            data2[:, yoffset:] = np.nan
            data2[xoffset:, :yoffset] = data1[:-xoffset, -yoffset:]
        elif xoffset < 0 and yoffset > 0:
            data2[xoffset:, :] = np.nan
            data2[:, :yoffset] = np.nan
            data2[:xoffset, yoffset:] = data1[-xoffset:, :-yoffset]
        output_dataset.GetRasterBand(i+1).WriteArray(data2)

def sortXYListforTrain(date, path, keyWord):
    ds2 = gdal.Open(path + 'data_tmp/' + date +'_seg.tif', GA_ReadOnly)
    segMap = ds2.GetRasterBand(1).ReadAsArray()
    corrList = np.loadtxt(path + 'data_tmp/' + keyWord + '_'+ date + '_seg.txt')
    xyList = []
    print('Number of samples:   ', len(corrList))
    for i in range(len(corrList)):
        xyList.append(np.where(segMap == corrList[i]))
    with open(path + 'data_tmp/' + keyWord +"_XYList_"+date, "wb") as fp:
        pickle.dump(xyList, fp)
    print('XyList sorted!')

def sortXYListforPredict(date, path):
    ds2 = gdal.Open(path + 'data_tmp/' + date +'_seg.tif', GA_ReadOnly)
    segMap = ds2.GetRasterBand(1).ReadAsArray()
    xyList = []

    for i in range(int(np.max(np.unique(segMap))+1)):
        xyList.append(np.where(segMap == i))
        print(i, '    done!')
    with open(path + 'data_tmp/' + "Predict_XYList_"+date, "wb") as fp:
        pickle.dump(xyList, fp)
    print('Predict xyList sorted!')


def extractDataset(date, path, keyWord, dateList, fusionFlag):
    from skimage.feature import greycomatrix, greycoprops
    ds = gdal.Open(path+'data_tmp/dataWrappedForTrain_'+date+'.tif', GA_ReadOnly)

    with open(path + 'data_tmp/' + keyWord + "_XYList_"+date, "rb") as fp:
        xyList = pickle.load(fp)
    print('Number of samples:    ', len(xyList))
    sampleNum = 29
    if fusionFlag == 0:
        print('No data fusion')
        trainingDataset = np.array([])
        for i in range(len(xyList)):
            trainingDataset = np.append(trainingDataset, len(xyList[i][0]))  # append pixel number
        trainingDataset.tofile(path + 'data_tmp/tmp/' + keyWord + '_PixelNumber'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset = np.array([])
        print('pixel num done!')


        for j in range(7):  # blue/green/red/rededge/nir/temperature/DSM: mean and std
            trainingDataset1 = np.array([])
            trainingDataset2 = np.array([])
            tmpData = ds.GetRasterBand(j + 1).ReadAsArray()
            for i in range(len(xyList)):
                trainingDataset1 = np.append(trainingDataset1, np.nanmean(tmpData[xyList[i]]))
                trainingDataset2 = np.append(trainingDataset2, np.nanstd(tmpData[xyList[i]]))
            trainingDataset1.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(j)+'_mean'+date+'_'+str(fusionFlag)+'.bin')
            trainingDataset2.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(j)+'_std'+date+'_'+str(fusionFlag)+'.bin')
            trainingDataset1 = np.array([])
            trainingDataset2 = np.array([])
            print('index '+str(j)+'  done!')

        #ndvi
        tmpData = (ds.GetRasterBand(5).ReadAsArray()-ds.GetRasterBand(3).ReadAsArray())/(ds.GetRasterBand(5).ReadAsArray()+ds.GetRasterBand(3).ReadAsArray())
        for i in range(len(xyList)):
            trainingDataset1 = np.append(trainingDataset1, np.nanmean(tmpData[xyList[i]]))
            trainingDataset2 = np.append(trainingDataset2, np.nanstd(tmpData[xyList[i]]))
        trainingDataset1.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index7_mean'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset2.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index7_std'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset1 = np.array([])
        trainingDataset2 = np.array([])
        print('index 7  done!')

        #blue-ndvi
        tmpData = (ds.GetRasterBand(5).ReadAsArray()-ds.GetRasterBand(1).ReadAsArray())/(ds.GetRasterBand(5).ReadAsArray()+ds.GetRasterBand(1).ReadAsArray())
        for i in range(len(xyList)):
            trainingDataset1 = np.append(trainingDataset1, np.nanmean(tmpData[xyList[i]]))
            trainingDataset2 = np.append(trainingDataset2, np.nanstd(tmpData[xyList[i]]))
        trainingDataset1.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index8_mean'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset2.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index8_std'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset1 = np.array([])
        trainingDataset2 = np.array([])
        print('index 8  done!')

        #green-ndvi
        tmpData = (ds.GetRasterBand(5).ReadAsArray()-ds.GetRasterBand(2).ReadAsArray())/(ds.GetRasterBand(5).ReadAsArray()+ds.GetRasterBand(2).ReadAsArray())
        for i in range(len(xyList)):
            trainingDataset1 = np.append(trainingDataset1, np.nanmean(tmpData[xyList[i]]))
            trainingDataset2 = np.append(trainingDataset2, np.nanstd(tmpData[xyList[i]]))
        trainingDataset1.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index9_mean'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset2.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index9_std'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset1 = np.array([])
        trainingDataset2 = np.array([])
        print('index 9  done!')

        #rededge-ndvi
        tmpData = (ds.GetRasterBand(5).ReadAsArray()-ds.GetRasterBand(4).ReadAsArray())/(ds.GetRasterBand(5).ReadAsArray()+ds.GetRasterBand(4).ReadAsArray())
        for i in range(len(xyList)):
            trainingDataset1 = np.append(trainingDataset1, np.nanmean(tmpData[xyList[i]]))
            trainingDataset2 = np.append(trainingDataset2, np.nanstd(tmpData[xyList[i]]))
        trainingDataset1.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index10_mean'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset2.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index10_std'+date+'_'+str(fusionFlag)+'.bin')
        trainingDataset1 = np.array([])
        trainingDataset2 = np.array([])
        print('index 10  done!')


        #calculate the maximum gray level
        tmpData = (ds.GetRasterBand(1).ReadAsArray()+ds.GetRasterBand(2).ReadAsArray()+ds.GetRasterBand(3).ReadAsArray())/3
        grayMax = np.nanmax(tmpData)
        grayMin = np.nanmin(tmpData)
        for i in range(len(xyList)):
            xmax = np.max(xyList[i][0])
            xmin = np.min(xyList[i][0])
            ymax = np.max(xyList[i][1])
            ymin = np.min(xyList[i][1])
            tmpDataset_GLCM = np.full((xmax - xmin + 1, ymax - ymin + 1), np.nan)
            tmpDataset_GLCM[xyList[i][0]-xmin, xyList[i][1]-ymin] = tmpData[xyList[i]]
            tmpDataset_GLCM = (tmpDataset_GLCM - grayMin) * 255 / (grayMax - grayMin)
            tmpDataset_GLCM[tmpDataset_GLCM == 0] = 1.0
            tmpDataset_GLCM[np.isnan(tmpDataset_GLCM)] = 0.0
            tmpDataset_GLCM = tmpDataset_GLCM.astype(int)

            GLCM_tmp = greycomatrix(tmpDataset_GLCM, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True,
                                    normed=True)
            GLCM_tmp = GLCM_tmp[1:, 1:, :, :]
            GLCM = np.ones([np.shape(GLCM_tmp)[0], np.shape(GLCM_tmp)[1], np.shape(GLCM_tmp)[2], 1])
            GLCM[:, :, :, 0] = np.mean(GLCM_tmp, axis=3)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                temp = greycoprops(GLCM, prop)
                trainingDataset = np.append(trainingDataset, temp)

        trainingDataset = trainingDataset.reshape(-1,6).T.flatten()
        trainingDataset.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index11'+date+'_'+str(fusionFlag)+'.bin')
        print('index 11  done!')

    # Fusion from other flights?
    elif fusionFlag == 1:
        trainingDataset = np.array([])

        gg = np.where(dateList == date)
        dateList = dateList[:gg[0][0]+1]
        print('Data fusion from NDVI time series:     ', dateList)
        sampleNum += len(dateList)+1
        for i in range(len(dateList) - 1):
            date1 = dateList[i]
            date2 = dateList[i + 1]
            if i == 0:
                ds1 = gdal.Open(path + 'data_tmp/dataNDVI_' + date1 + '.tif', GA_ReadOnly)
                band = ds1.GetRasterBand(1)
                sum = ds1.GetRasterBand(1).ReadAsArray()
                maxID = np.full((band.YSize, band.XSize), 0)
                maxRaster = ds1.GetRasterBand(1).ReadAsArray()

            ds1 = gdal.Open(path + 'data_tmp/dataNDVI_' + date1 + '.tif', GA_ReadOnly)
            ds2 = gdal.Open(path + 'data_tmp/dataNDVI_' + date2 + '.tif', GA_ReadOnly)
            data2 = ds2.GetRasterBand(1).ReadAsArray()
            data1 = ds1.GetRasterBand(1).ReadAsArray()
            sum += ds2.GetRasterBand(1).ReadAsArray()
            maxID[data2 > maxRaster] = i + 1
            maxRaster[data2 > maxRaster] = data2[data2 > maxRaster]
            tmpData = ds2.GetRasterBand(1).ReadAsArray() - ds1.GetRasterBand(1).ReadAsArray()
            for j in range(len(xyList)):
                trainingDataset = np.append(trainingDataset, np.nanmean(tmpData[xyList[j]]))
        tmpData = sum/len(dateList)
        for j in range(len(xyList)):
            trainingDataset = np.append(trainingDataset, np.nanmean(tmpData[xyList[j]]))
        tmpData = maxID
        for j in range(len(xyList)):
            trainingDataset = np.append(trainingDataset, np.bincount(tmpData[xyList[j]]).argmax())

        trainingDataset.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index12_'+date+'_'+str(fusionFlag)+'.bin')
        print('index 12 done!')

    elif fusionFlag == 2:
        trainingDataset = np.array([])
        print('Data fusion from NDVI time series:     ', dateList)
        sampleNum += len(dateList)+1
        for i in range(len(dateList) - 1):
            date1 = dateList[i]
            date2 = dateList[i + 1]
            if i == 0:
                ds1 = gdal.Open(path + 'data_tmp/dataNDVI_' + date1 + '.tif', GA_ReadOnly)
                band = ds1.GetRasterBand(1)
                sum = ds1.GetRasterBand(1).ReadAsArray()
                maxID = np.full((band.YSize, band.XSize), 0)
                maxRaster = ds1.GetRasterBand(1).ReadAsArray()

            ds1 = gdal.Open(path + 'data_tmp/dataNDVI_' + date1 + '.tif', GA_ReadOnly)
            ds2 = gdal.Open(path + 'data_tmp/dataNDVI_' + date2 + '.tif', GA_ReadOnly)
            data2 = ds2.GetRasterBand(1).ReadAsArray()
            sum += ds2.GetRasterBand(1).ReadAsArray()
            maxID[data2 > maxRaster] = i + 1
            maxRaster[data2 > maxRaster] = data2[data2 > maxRaster]
            tmpData = ds2.GetRasterBand(1).ReadAsArray() - ds1.GetRasterBand(1).ReadAsArray()
            for j in range(len(xyList)):
                trainingDataset = np.append(trainingDataset, np.nanmean(tmpData[xyList[j]]))
        tmpData = sum/len(dateList)
        for j in range(len(xyList)):
            trainingDataset = np.append(trainingDataset, np.nanmean(tmpData[xyList[j]]))
        tmpData = maxID
        for j in range(len(xyList)):
            trainingDataset = np.append(trainingDataset, np.bincount(tmpData[xyList[j]]).argmax())

        trainingDataset.tofile(path + 'data_tmp/tmp/' + keyWord+'_Index12_'+date+'_'+str(fusionFlag)+'.bin')
        print('index 12 done!')
    else:
        print('Please sepcify the correct fusionFlag          !!!')
    np.savetxt(path + 'data_tmp/sampleNum_'+date+'_'+str(fusionFlag)+'.txt', np.array([sampleNum]))


def conbineDataset(date, path, keyWord, dateList, fusionFlag):
    if fusionFlag == 0:
        tmpData = np.fromfile(path + 'data_tmp/tmp/' + keyWord+'_PixelNumber'+date+'_'+'0.bin')
        for i in range(11):
            tmpData = np.append(tmpData, np.fromfile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_mean'+date+'_'+'0.bin'))
            tmpData = np.append(tmpData, np.fromfile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_std'+date+'_'+'0.bin'))
        for i in [11]:
            tmpData = np.append(tmpData, np.fromfile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+date + '_' +'0.bin'))
    elif fusionFlag == 1 or fusionFlag==2:
        tmpData = np.fromfile(path + 'data_tmp/'+ keyWord+'_dataset_'+date+'_0.bin')
        for i in [12]:
            tmpData = np.append(tmpData, np.fromfile(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_'+date+'_' + str(fusionFlag)+'.bin'))

    tmpData.tofile(path + 'data_tmp/'+ keyWord+'_dataset_'+date+'_'+str(fusionFlag)+'.bin')
    Xnum = np.loadtxt(path + 'data_tmp/sampleNum_'+date+'_'+str(fusionFlag)+'.txt')
    np.savetxt(path + 'data_tmp/'+ keyWord+'_dataset_'+date+'_'+str(fusionFlag)+'.txt', tmpData.reshape(int(Xnum),-1).T)


def DeleteDatesetTMP(date, path, keyWord, dateList, fusionFlag):
    os.remove(path + 'data_tmp/tmp/' + keyWord+'_PixelNumber'+date+'_'+'0.bin')
    for i in range(11):
        os.remove(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_mean'+date+'_'+'0.bin')
        os.remove(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_std'+date+'_'+'0.bin')
    for i in [11]:
        os.remove(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+date + '_' +'0.bin')
    if fusionFlag == 1 or fusionFlag==2:
        for i in [12,13]:
            os.remove(path + 'data_tmp/tmp/' + keyWord+'_Index'+str(i)+'_'+date+'_' + str(fusionFlag)+'.bin')



def TrainAndValid(date, path, fusionFlag, n_estimators, iterNum):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.model_selection import cross_val_score, KFold
    import seaborn
    import joblib
    Xnum = np.loadtxt(path + 'data_tmp/sampleNum_'+date+'_'+str(fusionFlag)+'.txt')
    X = np.fromfile(path+'data_tmp/'+'All_dataset_'+date+'_'+str(fusionFlag)+'.bin').reshape(int(Xnum),-1).T[:,1:]
    Y = np.loadtxt(path + 'data_tmp/' 'All_'+date+'.txt', skiprows=1, delimiter=',').reshape(-1,4)[:,1]

    index = np.arange(np.shape(Y)[0])
    np.random.shuffle(index)
    X = X[index, :]
    Y = Y[index]

    Y = np.delete(Y, np.where(np.isnan(X))[0], axis=0)
    X = np.delete(X, np.where(np.isnan(X))[0], axis=0)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=20,bootstrap=True, n_jobs=1)

    n_split = 5
    iterNum = iterNum
    kf = KFold(n_splits=n_split)
    count = 0
    scoreList = np.full((n_split * iterNum, 15), -9999.0)
    bandImportanceList = np.full((n_split * iterNum, 36), -9999.0)

    for iter in range(iterNum):
        for train_index, test_index in kf.split(X):
            train_X, train_Y = X[train_index], Y[train_index]
            test_X, test_Y = X[test_index], Y[test_index]
            rf.fit(train_X, train_Y)
            cm = confusion_matrix(test_Y, rf.predict(test_X))
            IDList = np.unique(train_Y)
            for xx in range(len(IDList)):
                scoreList[int(count), int(IDList[xx])] = cm[xx, xx] / np.sum(cm[:, xx])
            scoreList[count, -1] = rf.score(test_X, test_Y)
            joblib.dump(rf, path + 'data_tmp/' + date + '_' + str(fusionFlag) + str(count) + '.model')
            if fusionFlag==1:
                bandImportanceList[count, :len(rf.feature_importances_)-2] = rf.feature_importances_[:-2]
                #bandImportanceList[count, -2:] = rf.feature_importances_[-2:]
            else:
                bandImportanceList[count, :len(rf.feature_importances_)] = rf.feature_importances_[:]
            print(fusionFlag, count, np.shape(scoreList), np.shape(bandImportanceList))
            count = count + 1
    #print(scoreList)
    np.savetxt(path + 'score' + date + '_' + str(fusionFlag) + '.txt', scoreList)
    np.savetxt(path + 'bandImportance_' + date + '_' + str(fusionFlag) + '.txt', bandImportanceList)

    print('The beat performed model is:     ', np.where(scoreList == np.max(scoreList[:,-1]))[0][0])
    shutil.copy(
        path + 'data_tmp/' + date + '_' + str(fusionFlag) + str(
            np.where(scoreList == np.max(scoreList))[0][0]) + '.model', \
        path + 'data_tmp/' + date + '_' + str(fusionFlag) + '.model')
    for j in range(count):
        os.remove(path + 'data_tmp/' + date + '_' + str(fusionFlag) + str(j) + '.model')




    #scores = cross_val_score(rf, X, Y, cv=5)
    #print(scores, np.mean(scores))


def Predict(date, path, fusionFlag):
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    import seaborn
    import joblib

    Xnum = np.loadtxt(path + 'data_tmp/sampleNum_'+date+'_'+str(fusionFlag)+'.txt')
    X = np.fromfile(path+'data_tmp/'+'Predict_dataset_'+date+'_'+str(fusionFlag)+'.bin').reshape(int(Xnum),-1).T[:,1:]
    print('Num of data points containing NaN:    ', len(np.where(np.isnan(X))[0]))
    #X = np.delete(X, np.where(np.isnan(X))[0], axis=0)
    X[np.where(np.isnan(X))] = 0.1

    print("Shape of final input datasets:     ", np.shape(X))
    print('Predicting ...')
    rf = joblib.load(path + 'data_tmp/' + date + '_' + str(fusionFlag) + '.model')
    predictResults = rf.predict(X)
    print('Prediction done!')

    with open(path + 'data_tmp/' "Predict_XYList_"+date, "rb") as fp:
        xyList = pickle.load(fp)

    dataset = gdal.Open(path + 'data_tmp/dataWrappedForTrain_'+date+'.tif', GA_ReadOnly)
    band = dataset.GetRasterBand(1)

    file_driver = gdal.GetDriverByName('Gtiff')
    if os.path.exists(path + 'Predict_'+date+'_'+str(fusionFlag)+'.tif'):
        os.remove(path + 'Predict_'+date+'_'+str(fusionFlag)+'.tif')
    output_dataset = file_driver.Create(path + 'Predict_'+date+'_'+str(fusionFlag)+'.tif', band.XSize, band.YSize, 1, band.DataType)
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    data = np.full((band.YSize, band.XSize), np.nan)

    for i in range(len(xyList)):
        data[xyList[i]] = predictResults[i]

    for i in (np.where(np.isnan(X))[0]):
        data[xyList[i]] = np.nan

    print('read rasterMask...')
    ds2 = gdal.Open(path + 'data_tmp/GPS/rasterMask.tif', GA_ReadOnly)
    rasterMask= ds2.GetRasterBand(1).ReadAsArray()
    data[rasterMask==0] = np.nan
    print('read streamArea...')
    ds2 = gdal.Open(path + 'data_tmp/GPS/streamArea.tif', GA_ReadOnly)
    rasterMask= ds2.GetRasterBand(1).ReadAsArray()
    data[(rasterMask==1)&(data==13)] = 3
    print('read northArea...')
    if date!='20210330' and date!='20210426':
        ds2 = gdal.Open(path + 'data_tmp/GPS/northArea.tif', GA_ReadOnly)
        rasterMask= ds2.GetRasterBand(1).ReadAsArray()
        data[(rasterMask==0)&(data==1)] = 2
        data[(rasterMask==0)&(data==6)] = 2
    print('write prediction map...')
    output_dataset.GetRasterBand(1).WriteArray(data)



def visiliseMaps(date, path, cbFlag, fusionFlag):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    ds = gdal.Open(path + 'Predict_'+date+'_'+str(fusionFlag)+'.tif', GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray()

    fig, ax = plt.subplots(figsize=(6, 6),dpi=600)
    fig.subplots_adjust(bottom=0.1, top=1, left=0, right=1)

    cmap = mpl.colors.ListedColormap([(148/255, 49/255, 38/255),(99/255, 57/255, 116/255),(212/255, 172/255, 13/255),\
                                      (17/255, 120/255, 100/255),(236/255, 112/255, 99/255),(151/255, 154/255, 154/255),\
                                      (230/255, 126/255, 34/255),(171/255, 235/255, 198/255),(249/255, 231/255, 159/255),\
                                      (23/255, 32/255, 42/255),(25/255, 111/255, 61/255),(46/255, 134/255, 193/255),\
                                      (128/255, 139/255, 150/255),(84/255, 153/255, 199/255)])
    cmap.set_over('0.25')
    #cmap.set_under('0.75')

    #cmap = mpl.cm.viridis
    indexNum = 14
    bounds = np.arange(indexNum+1)+0.5
    bounds = bounds.tolist()
    tick   = np.arange(indexNum+1)
    tick   = tick.tolist()

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if cbFlag:
        labelList = ['Fescue ', 'Chee Grass', 'Ryegrass',\
                     'Reed', 'Nut Grass', 'Senesced grass',\
                     'Nettle', 'Growing grass', 'Mixed G/S grass',\
                     'Shadow','Forest','Water','Water with vegtation cover', 'Oxalis']
        iList = [7,8,5,10,11,9,0,1,2,3,4,6,13]
        for i in iList:
            plt.plot(0, 0, "-", color=cmap(i/indexNum), label=labelList[i], linewidth = 10)
        plt.legend(loc="upper right", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
        plt.legend(frameon=False)
        plt.axis('off')
        plt.savefig(path + 'vegMap_legend.png', transparent=True)
        os._exit(0)

    plt.imshow(data, cmap = cmap, norm = norm)
    plt.axis('off')
    plt.savefig(path+'vegMap_'+date+'_'+str(fusionFlag)+'.png', transparent=True)


def visilisePlots(dateList, path, fusionFlag, iterNum):
    import seaborn as sns
    import matplotlib as mpl
    n_split = 5
    scoreList = np.full((len(dateList), 15),-9999.0)
    bandImportance = np.full((len(dateList), 36),-9999.0)
    for i in range(len(dateList)):
        scoreList_tmp = np.loadtxt(path + 'score' + dateList[i] + '_' + str(fusionFlag) + '.txt').flatten().reshape(iterNum*n_split,-1)
        bandImportance_tmp = np.loadtxt(path + 'bandImportance_' + dateList[i] + '_' + str(fusionFlag) + '.txt').flatten().reshape(iterNum*n_split,-1)
        scoreList[i,:] = np.nanmean(scoreList_tmp, axis=0)
        bandImportance[i,:] = np.nanmean(bandImportance_tmp, axis=0)
    scoreList[scoreList==-9999] = np.nan
    bandImportance[bandImportance==-9999] = np.nan
    IDList = [7, 8, 5, 10, 11, 12, 9, 0, 1, 2, 3, 4, 6, 13,14]
    scoreList_final = np.array([])
    for j in range(len(IDList)):
        scoreList_final = scoreList[:,IDList]
    fig = plt.figure(figsize=(3, 3))
    ax = sns.heatmap(scoreList_final.T, vmin=0.7, vmax=1.0, cmap='cividis', cbar=False)
    plt.axis('off')
    plt.savefig(path+'plots/accuracy_'+str(fusionFlag)+'.png', transparent=True, dpi=200)

    fig = plt.figure(figsize=(3, 6))
    ax = sns.heatmap(bandImportance.T, vmin=0, vmax=0.15, cmap='cividis', cbar=False)
    plt.axis('off')
    plt.savefig(path+'plots/bandImportance_'+str(fusionFlag)+'.png', transparent=True, dpi=200)


def updateAllMaps(dateList, path):
    import matplotlib as mpl

    cmap = mpl.colors.ListedColormap([(148/255, 49/255, 38/255),(99/255, 57/255, 116/255),(212/255, 172/255, 13/255),\
                                      (17/255, 120/255, 100/255),(236/255, 112/255, 99/255),(151/255, 154/255, 154/255),\
                                      (230/255, 126/255, 34/255),(171/255, 235/255, 198/255),(249/255, 231/255, 159/255),\
                                      (23/255, 32/255, 42/255),(25/255, 111/255, 61/255),(46/255, 134/255, 193/255),\
                                      (128/255, 139/255, 150/255),(84/255, 153/255, 199/255)])
    cmap.set_over('0.25')
    indexNum = 14
    bounds = np.arange(indexNum+1)+0.5
    bounds = bounds.tolist()
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    count = 1

    fig = plt.figure(figsize=(len(dateList)*7,3*10),dpi=200)


    for fusionFlag in range(3):
        for i in range(len(dateList)):
            ds = gdal.Open(path + 'Predict_' + dateList[i] + '_' + str(fusionFlag) + '.tif', GA_ReadOnly)
            data = ds.GetRasterBand(1).ReadAsArray()
            ax = fig.add_subplot(3, len(dateList), count)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.axis('off')
            count += 1
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
    plt.savefig('aaa.png', transparent=True)

