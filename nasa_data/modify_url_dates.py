from datetime import datetime, timedelta

# Define the date ranges to iterate over
date_ranges = [
    ("2015-12-01", "2015-12-13"),
    ("2018-09-04", "2018-09-16"),
    ("2020-10-25", "2020-11-06"),
    ("2024-07-22", "2024-08-03")
]

# Base URL structure
base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.07/{year}/{month:02d}/3B-DAY.MS.MRG.3IMERG.{date}-S000000-E235959.V07B.nc4.nc4?MWprecipitation[0:0][3008:3010][1043:1046],precipitation_cnt_cond[0:0][3008:3010][1043:1046],precipitation[0:0][3008:3010][1043:1046],MWprecipitation_cnt[0:0][3008:3010][1043:1046],MWprecipitation_cnt_cond[0:0][3008:3010][1043:1046],probabilityLiquidPrecipitation[0:0][3008:3010][1043:1046],time_bnds[0:0][0:1],precipitation_cnt[0:0][3008:3010][1043:1046],time,lon[3008:3010],lat[1043:1046],nv"

# Function to generate URLs for each day within the specified date range
def generate_urls(start_date, end_date):
    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        url = base_url.format(
            year=current_date.year,
            month=current_date.month,
            date=date_str
        )
        urls.append(url)
        current_date += timedelta(days=1)
    return urls

# Parse date ranges and generate URLs
all_urls = []
for start_str, end_str in date_ranges:
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    all_urls.extend(generate_urls(start_date, end_date))

# Display the generated URLs
for url in all_urls:
    print(url)

# Save the generated URLs to a text file
output_file = '../nasa_data/newly_generated_urls.txt'
with open(output_file, "w") as f:
    for url in all_urls:
        f.write(url + "\n")

print(f"All URLs saved to {output_file}")
