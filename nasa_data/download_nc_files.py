import re
import requests
import os


NC_FILE_LIST = 'nasa_data/subset_GPM_3IMERGDF_07_20241021_040313_.txt'
DOWNLOAD_DIR = 'nasa_data/precipitation_data'

with open (NC_FILE_LIST, 'r') as f:
    nc_files = f.readlines()

nc_files = [f.strip() for f in nc_files[2:]]

for file_link in nc_files:
    # get the date from the file name using regex
    date = re.search(r'\d{8}', file_link).group()
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    file_name = f'{year}-{month}-{day}.nc4'
    file_path = os.path.join(DOWNLOAD_DIR, file_name)

    # verify if file already exists
    if os.path.exists(file_path):
        print(f"File {file_name} already exists. Skipping download.")
        continue

    with requests.Session() as s:
        s.auth = (os.getenv('NASA_USER_NAME'), os.getenv('NASA_PASSWORD'))
        
        try:
            response = s.get(s.request('get', file_link).url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name}: {response.status_code}")
                with open("failed_downloads.log", "a") as log_file:
                    log_file.write(f"{file_link}\n")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            with open("failed_downloads.log", "a") as log_file:
                log_file.write(f"{file_link}\n")
