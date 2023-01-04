from PIL import Image
from os import listdir
from os.path import isfile, join
from collections import defaultdict

inputFiles = "d:/Dane/diabetic-retinopathy-detection/unpacked/train"
outputFile = "d:/Dane/diabetic-retinopathy-detection/unpacked/train_small_2"

onlyfiles = [(f, join(inputFiles, f), join(outputFile, f)) for f in listdir(inputFiles) if isfile(join(inputFiles, f))]
existingfiles = [join(outputFile, f) for f in listdir(outputFile) if isfile(join(outputFile, f))]
existingfilesSet = set(existingfiles)

filteredFiles = [x for x in onlyfiles if x[2] not in existingfilesSet]

def computeCropRectangle(input):
    print(input)
    (width, height, xMin, yMin, xMax, yMax) = input
    wi = xMax - xMin
    he = yMax - yMin
    
    if wi > he:
        if height > wi:
            he = wi
            yMin = (int)(((yMax + yMin) - he) / 2)
            yMax = yMin + he
            return (xMin, yMin, xMax, yMax)
        else:
            return (xMin, 0, xMax, height - 1)
    elif he > wi:
        if width > he:
            wi = he
            print(((xMax + xMin) - wi))
            xMin = (int)(((xMax + xMin) - wi) / 2)
            xMax = xMin + wi
            return (xMin, yMin, xMax, yMax)
        else:
            return (0, yMin, width - 1, yMax)
    else:
        return (xMin, yMin, xMax, yMax)


def isPixelBlack(width, height, pix, x, y, treshold):
    pixCount = 1 # the fist one is the one in the middle
    blackCount = 0

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            xx = x + dx
            yy = y + dy

            if xx >= 0 and yy >= 0 and xx < width and yy < height:
                pixel = pix[xx, yy]
                pixCount = pixCount + 1
                isBlack = pixel[0] < treshold and pixel[1] < treshold and pixel[2] < treshold
                if isBlack:
                    blackCount = blackCount + 1

    return blackCount > (pixCount / 2)


def isPixelLight(width, height, pix, x, y, treshold):
    pixCount = 1 # the fist one is the one in the middle
    lightCount = 0
    treshold = 255 - treshold

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            xx = x + dx
            yy = y + dy

            if xx >= 0 and yy >= 0 and xx < width and yy < height:
                pixel = pix[xx, yy]
                pixCount = pixCount + 1
                isLight = pixel[0] > treshold or pixel[1] > treshold or pixel[2] > treshold
                if isLight:
                    lightCount = lightCount + 1

    return lightCount > (pixCount * 3.0 / 4.0)    


def computeCropDataExt(im, pix, treshold, dontCheckLightPixels):
    xMin = 1000000000
    xMax = 0
    yMin = 1000000000
    yMax = 0
    width = im.size[0]
    height = im.size[1]
    
    allAreBlack = True

    for x in range(0, im.size[0]):
        for y in range(0, im.size[1]):
            pixel = pix[x,y]
            isBlack = pixel[0] < treshold and pixel[1] < treshold and pixel[2] < treshold
            
            if not isBlack:
                allAreBlack = False

                if dontCheckLightPixels:
                    lightResult = isPixelLight(width, height, pix, x, y, treshold)
                else:
                    lightResult = True

                if not isPixelBlack(width, height, pix, x, y, treshold) and lightResult:
                    if x < xMin:
                        xMin = x
                        
                    if x > xMax:
                        xMax = x
                        
                    if y < yMin:
                        yMin = y
                        
                    if y > yMax:
                        yMax = y

    if allAreBlack:
        return None

    cropRect = computeCropRectangle((width, height, xMin, yMin, xMax + 1, yMax + 1))
    
    #cropRect = (xMin, yMin, xMax + 1, yMax + 1)
    return cropRect             


def computeCropData(im, pix, treshold):
    xMin = 1000000000
    xMax = 0
    yMin = 1000000000
    yMax = 0
    width = im.size[0]
    height = im.size[1]
    
    allAreBlack = True

    for x in range(0, im.size[0]):
        for y in range(0, im.size[1]):
            pixel = pix[x,y]
            isBlack = pixel[0] < treshold and pixel[1] < treshold and pixel[2] < treshold
            
            if not isBlack:
                allAreBlack = False

                if x < xMin:
                    xMin = x
                    
                if x > xMax:
                    xMax = x
                    
                if y < yMin:
                    yMin = y
                    
                if y > yMax:
                    yMax = y

    if allAreBlack:
        return None

    cropRect = computeCropRectangle((width, height, xMin, yMin, xMax + 1, yMax + 1))
    
    #cropRect = (xMin, yMin, xMax + 1, yMax + 1)
    return cropRect        


def process_images(files, outputPath, existingfilesSet, useExtended, checkLightPixels):
    sizes = {}
    ix = 0
    for (fileName, pathIn, outFile) in files:
        print(pathIn)
        im = Image.open(pathIn)
        pix = im.load()
        x = im.size[0]
        y = im.size[1]
        si = (x,y)
        # if si in sizes:
        #     cropData = sizes[si]
        # else:
        #     cropData = computeCropData(im, pix, 30)
        #     sizes[si] = cropData

        #outFile = outputPath + "/" + fileName

        if outFile in existingfilesSet:
            print("File already exists " + outFile)
            continue

        if isfile(outFile):
            print("File already exists " + outFile)
            continue
        
        if useExtended:
            cropData = computeCropDataExt(im, pix, 50, checkLightPixels)
        else:
            cropData = computeCropData(im, pix, 50)

        if cropData is None:
            print("No crop data for image " + pathIn)
            continue

        imCrop = im.crop(cropData)
        #outFile = outputPath + "/" + fileName + '.png'
        
        imResize = imCrop.resize((224, 224))
        imResize.save(outFile)
        print(ix)
        print(outFile)
        print(si)
        print(cropData)
        ix = ix + 1

process_images(filteredFiles, outputFile, existingfilesSet, False, False)