import requests
from zipfile import ZipFile
import os
import gdown

def download_file_from_google_drive_with_gdown(id, destination):
    url = 'https://drive.google.com/uc?id=' + id
    gdown.download(url, destination, quiet=False)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def extract_files(name):
        
    Zip = ZipFile(name)
    Zip.extractall()
    
def remove_zip(name):
    os.remove(name)

    
    
##### Paderborn University Bearing Data #####
def PaderbornBearingData():
    
    data_id = '15v1fwFxfrntTE1FVdNvZXzMF5xufMa3Z'
    destination = 'PaderbornBearingData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)

##### Motor Temperature Data #####
def MotorTempData():
    

    data_id = '11Q5emmsc1dMMiMGe8niho5x4-3I5Oc_K'
    destination = 'MotorTempData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)
    
##### Turning Chatter Data #####
def ChatterData():
    

    data_id = '1z_2ceidvHmE5p7XCD4PaGn4ezvcZMxdD'
    destination = 'ChatterData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)

##### 3D printing Data #####
def ThreeDPrintingData():
    

    data_id = '1VhZcOgNOEw_Sciuww25XZdIuaqO90Nkj'
    destination = 'ThreeDPrintingData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)

##### Mercedes Green Manufacturing Data #####
def MercedesData():
    

    data_id = '1D7eQDV4h6lEXnNE1Cbk1kRU62Dn9xMnb'
    destination = 'MercedesData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)

##### Lithography Data #####
def LithographyData():
    

    data_id = '1XY4fbNtzKrXXPtfiPGpwunWvyIDH_V57'
    destination = 'LithographyData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)
    
##### Gearbox Data #####
def GearboxData():
    

    data_id = '1aTFu-M8V5CxbDY4e-nRBLNuPmbsbAbgk'
    destination = 'GearboxData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)
    
##### Casting Data #####
def CastingData():
    

    data_id = '1qNnLCcq1HlzS0WmOCRlJfNC9ZF26j_6f'
    destination = 'CastingData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)
    
##### CWRU Bearing Data #####
def CWRUBearingData():
    
    data_id = '1nUjYdpJkEmjJTzG0j8EBZ9vQul0sedqk'
    destination = 'CWRUBearingData.zip'

    download_file_from_google_drive_with_gdown(data_id, destination)
    extract_files(destination)
    remove_zip(destination)
    
##### 3D Spatter Data #####
# def SpatterData():
    

#     data_id = '17XyK0S7dO9RKhg4rqfveDPPOprZVE_f0'
#     destination = 'SpatterData.zip'

#     download_file_from_google_drive(data_id, destination)
#     extract_files(destination)
#     remove_zip(destination)

    
