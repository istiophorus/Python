from PIL import Image
import os
import shutil
import cv2
import numpy as np

def process_image(pathIn, pathOut, treshold, negative, condition):
    with Image.open(pathIn) as im:
        pix = im.load()
        #if negative:
        #    valueMin = 255
        #    valueMax = 0
        #else:
        valueMin = 0
        valueMax = 255

        for x in range(0, im.size[0]):
            for y in range(0, im.size[1]):
                imagePixel = pix[x,y]

                if type(imagePixel) is tuple:

                    if negative:
                        pixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2], 255);
                    else:
                        pixel = imagePixel

                    if condition == "or":
                        condiutionResult = pixel[0] >= treshold or pixel[1] >= treshold or pixel[2] >= treshold
                    else:
                        condiutionResult = pixel[0] >= treshold and pixel[1] >= treshold and pixel[2] >= treshold

                    if condiutionResult:
                        newPixel = (valueMax, valueMax, valueMax, 255)
                    else:
                        newPixel = (valueMin, valueMin, valueMin, 255)
                        
                    im.putpixel((x, y), newPixel)
                else:

                    if negative:
                        pixel = 255 - imagePixel
                    else:
                        pixel = imagePixel

                    if pixel >= treshold:
                        newPixel = valueMax
                    else:
                        newPixel = valueMin
                        
                    im.putpixel((x, y), newPixel)

        im.save(pathOut, compression='None')


def update_extension(fileName, targetExtension = None):
    if targetExtension is not None:
        fileNameWithoutExtension = os.path.splitext(fileName)[0]
        return fileNameWithoutExtension + targetExtension
    else:
        return fileName


def get_file_extension(fileName):
    parts = os.path.splitext(fileName)
    if len(parts) > 1:
        return parts[-1]
    else:
        return ""        


def process_files(inputPath, outputPath, treshold, negative, condition = "or", targetExtension = None):
    files = os.listdir(inputPath)
    for fileName in files:
        inputFile = inputPath + fileName
        if targetExtension is not None:
            fileNameWithoutExtension = os.path.splitext(fileName)[0]
            outputFile = outputPath + fileNameWithoutExtension + targetExtension
        else:
            outputFile = outputPath + fileName
        process_image(inputFile, outputFile, treshold, negative, condition)        


def resize_image(inputPath, outputPath):
    #print(inputPath)
    #print(outputPath)
    with Image.open(inputPath) as image:
        if image.size[0] != TARGET_IMAGE_SIZE[0] or image.size[1] != TARGET_IMAGE_SIZE[1]:
            with image.resize(TARGET_IMAGE_SIZE) as resized_image:
                resized_image.save(outputPath, compression='None')
        else:
            image.save(outputPath, compression='None')


def filter_images(inputPath, outputPath, targetExtension, convertFunction):
    files = os.listdir(inputPath)
    for fileName in files:
        inputFile = inputPath + fileName
        
        if targetExtension is not None:
            fileNameWithoutExtension = os.path.splitext(fileName)[0]
            outputFile = outputPath + fileNameWithoutExtension + targetExtension
        else:
            outputFile = outputPath + fileName
            
        convertFunction(inputFile, outputFile)            


def resize_images(inputPath, outputPath, targetExtension):
    filter_images(inputPath, outputPath, targetExtension, resize_image)        


def median_filter(pathIn, pathOut):
    image = cv2.imread(pathIn)
    img_median = cv2.medianBlur(image, 15)
    cv2.imwrite(pathOut, img_median)       


def no_transform(pathIn, pathOut):
    shutil.copyfile(pathIn, pathOut)    


def resize_all_images(inputFolder, outputFolder, targetExtension = None):
    inputFolderTemp = inputFolder
    outputFolderTemp = outputFolder
    resize_images(inputFolderTemp, outputFolderTemp, targetExtension)    


def create_target_folders(baseFolder, folders):
    if not os.path.exists(baseFolder):
        raise ValueError('Specified base path does not exist ' + baseFolder)
        
    if not os.path.isdir(baseFolder):
        raise ValueError('Specified path is not a folder ' + baseFolder)
    
    for folder in folders:
        fullPath = baseFolder + folder
        if os.path.exists(fullPath):
            if os.path.isdir(fullPath):
                print('Folder already exists ' + fullPath)
            else:
                raise ValueError('Path already exists but it is not a folder' + fullPath)
        else:
            print('Createg folder ' + fullPath)
            os.mkdir(fullPath)    


TARGET_IMAGE_SIZE = (565, 584)            

refugeSourceFolder = 'd:/Dane/Archive/REFUGE/'
#driveTargetFolder = 'd:/Dane/DRIVE/'
driveTargetFolder = 'd:/Dane/DRIVE_no_transform/'


targetFolders = [
    'test',
    'training',
    'test/1st_manual',
    'test/2nd_manual',
    'test/images_resized',
    'test/images_resized_processed',
    'test/images',
    'test/mask',
    'test/_temp_segments_',
    'training/1st_manual',
    'training/images_resized',
    'training/images_resized_processed',
    'training/images',
    'training/mask',
    'training/_temp_segments_',
    'training/_temp_cup_',
    'training/_temp_disk_',
    'test/_temp_cup_',
    'test/_temp_disk_',
    'test/_temp_mask_',
    'training/_temp_mask_'
]


def generate_train_masks(inputFolder, outputFolder):
    process_files(inputFolder, outputFolder, 20, False, targetExtension = ".gif")


def generate_optic_cup(inputFolder, outputFolder):
    process_files(inputFolder, outputFolder, 200, True, "or", targetExtension = ".gif")        


def generate_optic_disk(inputFolder, outputFolder):
    process_files(inputFolder, outputFolder, 100, True, "or", targetExtension = ".gif")    


def rename_files(srcFolder, dstFolder, names_map, suffix, targetExtension = None):
    files = os.listdir(srcFolder)
    for fileName in files:
        fileNameWithOutExtension = os.path.splitext(fileName)[0]
       
        if fileNameWithOutExtension in names_map.keys():
            src = srcFolder + fileName
            fileNameMapped = names_map[fileNameWithOutExtension]
            
            if targetExtension is not None:
                ext = targetExtension
            else:
                ext = get_file_extension(fileName)
            
            newFileName = fileNameMapped + suffix + ext
            
            dst = dstFolder + newFileName
            
            shutil.copyfile(src, dst)
        else:
            raise ValueError(fileName)    


def map_names(sourceFolder, start_index = 1):
    files = os.listdir(sourceFolder)
    result = {}
    index = start_index
    for fileName in files:
        indexText = str(index)
        if index < 10:
            indexText = "000" + indexText
        elif index < 100:
            indexText = "00" + indexText
        elif index < 1000:
            indexText = "0" + indexText
        index += 1
        
        fileNameWithOutExtension = os.path.splitext(fileName)[0]
        
        result[fileNameWithOutExtension] = indexText
    return (result, index)            


create_target_folders(driveTargetFolder, targetFolders)    


inputFolder = refugeSourceFolder + 'train/images/'
outputFolder = driveTargetFolder + 'training/images_resized/'


resize_all_images(inputFolder, outputFolder, ".tif")

inputFolder = refugeSourceFolder + 'test/images/'
outputFolder = driveTargetFolder + 'test/images_resized/'

resize_all_images(inputFolder, outputFolder, ".tif")

inputFolder = refugeSourceFolder + 'train/gts/'
outputFolder = driveTargetFolder + 'training/_temp_segments_/'

resize_all_images(inputFolder, outputFolder, ".tif")

inputFolder = refugeSourceFolder + 'test/gts/'
outputFolder = driveTargetFolder + 'test/_temp_segments_/'

resize_all_images(inputFolder, outputFolder, ".tif")

inputFolder = driveTargetFolder + 'test/images_resized/'
outputFolder = driveTargetFolder + 'test/images_resized_processed/'

#filter_images(inputFolder, outputFolder, None, median_filter)
filter_images(inputFolder, outputFolder, None, no_transform)

inputFolder = driveTargetFolder + 'training/images_resized/'
outputFolder = driveTargetFolder + 'training/images_resized_processed/'

#filter_images(inputFolder, outputFolder, None, median_filter)
filter_images(inputFolder, outputFolder, None, no_transform)

inputFolder = driveTargetFolder + 'training/images_resized_processed/'
outputFolder = driveTargetFolder + 'training/_temp_mask_/'

generate_train_masks(inputFolder, outputFolder)

inputFolder = driveTargetFolder + 'test/images_resized_processed/'
outputFolder = driveTargetFolder + 'test/_temp_mask_/'

generate_train_masks(inputFolder, outputFolder)

generate_optic_cup(driveTargetFolder + 'training/_temp_segments_/', driveTargetFolder + 'training/_temp_cup_/')

generate_optic_cup(driveTargetFolder + 'test/_temp_segments_/', driveTargetFolder + 'test/_temp_cup_/')

generate_optic_disk(driveTargetFolder + 'training/_temp_segments_/', driveTargetFolder + 'training/_temp_disk_/')

generate_optic_disk(driveTargetFolder + 'test/_temp_segments_/', driveTargetFolder + 'test/_temp_disk_/')

(names_map, last_index) = map_names(driveTargetFolder + "training/images_resized_processed/")

print(last_index)

rename_files(
    driveTargetFolder + 'training/images_resized_processed/', 
    driveTargetFolder + 'training/images/',
    names_map,
    "_training", None)

rename_files(
    driveTargetFolder + "training/_temp_mask_/", 
    driveTargetFolder + "training/mask/",
    names_map,
    "_training_mask", None)    

rename_files(
    driveTargetFolder + 'training/_temp_disk_/', 
    driveTargetFolder + 'training/1st_manual/',
    names_map,
    "_manual1", None)    

(test_names_map, next_last_index) = map_names(driveTargetFolder + "test/images_resized_processed/", 401)    

print(next_last_index)

rename_files(
    driveTargetFolder + "test/images_resized_processed/", 
    driveTargetFolder + "test/images/",
    test_names_map,
    "_test", None)

rename_files(
    driveTargetFolder + "test/_temp_mask_/", 
    driveTargetFolder + "test/mask/",
    test_names_map,
    "_test_mask", None)    

rename_files(
    driveTargetFolder + "test/_temp_disk_/", 
    driveTargetFolder + "test/1st_manual/",
    test_names_map,
    "_manual1", None)    

rename_files(
    driveTargetFolder + "test/_temp_disk_/", 
    driveTargetFolder + "test/2nd_manual/",
    test_names_map,
    "_manual2", None)    