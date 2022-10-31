import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from ocr_license_plate import get_number
import os
import datetime

#vehicle class
class_dict = {'Daiatsu_Core': 0,
 'Daiatsu_Hijet': 1,
 'Daiatsu_Mira': 2,
 'FAW_V2': 3,
 'FAW_XPV': 4,
 'Honda_BRV': 5,
 'Honda_City_aspire': 6,
 'Honda_Grace': 7,
 'Honda_Vezell': 8,
 'Honda_city_1994': 9,
 'Honda_city_2000': 10,
 'Honda_civic_1994': 11,
 'Honda_civic_2005': 12,
 'Honda_civic_2007': 13,
 'Honda_civic_2015': 14,
 'Honda_civic_2018': 15,
 'KIA_Sportage': 16,
 'Suzuki_Every': 17,
 'Suzuki_Mehran': 18,
 'Suzuki_alto_2007': 19,
 'Suzuki_alto_2019': 20,
 'Suzuki_alto_japan_2010': 21,
 'Suzuki_carry': 22,
 'Suzuki_cultus_2018': 23,
 'Suzuki_cultus_2019': 24,
 'Suzuki_highroof': 25,
 'Suzuki_kyber': 26,
 'Suzuki_liana': 27,
 'Suzuki_margala': 28,
 'Suzuki_swift': 29,
 'Suzuki_wagonR_2015': 30,
 'Toyota HIACE 2000': 31,
 'Toyota_Aqua': 32,
 'Toyota_Hiace_2012': 33,
 'Toyota_Landcruser': 34,
 'Toyota_Passo': 35,
 'Toyota_Prado': 36,
 'Toyota_Vigo': 37,
 'Toyota_Vitz': 38,
 'Toyota_Vitz_2010': 39,
 'Toyota_axio': 40,
 'Toyota_corolla_2000': 41,
 'Toyota_corolla_2007': 42,
 'Toyota_corolla_2011': 43,
 'Toyota_corolla_2016': 44,
 'Toyota_fortuner': 45,
 'Toyota_pirus': 46,
 'Toyota_premio': 47}

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255.
    new_array = cv2.resize(img_array, (224, 224))
    return new_array.reshape(-1, 224, 224, 3)

def vehicle_class_finder(val, dict=class_dict):
  for k,v in dict.items():
    if(int(val) == int(v)):
    #   print(f'Vehicle is : {k}')
        return k

def model_detect(image, model):
    pred = model.predict([image])
    vehicle = vehicle_class_finder(np.argmax(pred))
    return vehicle

make_model_detection_model = load_model('model/vgg16_sf.h5')
# model.summary()

# prediction = model.predict([prepare("car_img/t101.png")])
# vehicle_class_finder(np.argmax(prediction))

# prediction = model.predict([prepare("car_img/t101.png")])
# vehicle_class_finder(np.argmax(prediction))
def engine():
  img_path = "./uploads/temp_image"
  # model_path = "model/vgg16_sf.h5"
   
  response = {
      'error': None,
      'Vehicle': None,
      'numplate': None,
      'time': None
  }

  path_list = []
  paths = os.listdir(img_path)

  for file in paths:
    path_list.append(os.path.join(img_path, file))
    
  for path in path_list:
    # print(path)
    img_224 = prepare(path)
    vehicle_model = model_detect(img_224, make_model_detection_model)
    num_plate = get_number(path)
    num_plate = num_plate.replace("\n", "")  

  response['error'] = 0
  response['Vehicle'] =  vehicle_model
  response['numplate'] = num_plate
  response['time'] = datetime.datetime.now()

  return response

    


