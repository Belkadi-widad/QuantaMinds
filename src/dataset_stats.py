import os
import mgrs
from geopy.geocoders import Nominatim
import re
import os
import re
import glob
import copy


directory = './data/DataTrain/input_tiles/'


def extend_dict(dict_1, dict_2):
    dict_1_copy = copy.deepcopy(dict_1)
    for key in dict_1:
        dict_1_copy[key].update(dict_2)
    return dict_1_copy


def merge_dicts(dict1, dict2): 
    # Initialize a new dictionary to hold the merged data
    merged_dict = {}

    # Iterate over the keys of the first dictionary
    for station in dict1:
        if station in dict2:
            # Combine the data from both dictionaries
            merged_dict[station] = {**dict1[station], **dict2[station]}
    return merged_dict

def extract_station(image_name):
    # Regular expression to match the localization part of the filename
    match = re.search(r'_T(\d{2}[A-Z]{3})_', image_name)
    if match:
        return match.group(1)
    else:
        return None

def get_lat_long_from_station(mgrs_tile):
    # Initialize MGRS converter
    mgrs_converter = mgrs.MGRS()
    # S2A_MSIL1C_20171008T064011_N0205_R120_T41SLA_20171008T064617_P933_0
    # MGRS tile identifier
    # Convert MGRS to latitude and longitude
    lat_lon = mgrs_converter.toLatLon(mgrs_tile)
    # lat_lon_str = ', '.join(str(l) for l in lat_lon)
    return lat_lon


def get_state_country(lat_lon):
    geoLoc = Nominatim(user_agent="GetLoc")
    # passing the coordinates
    locname = geoLoc.reverse(lat_lon)
    # printing the address/location name
    address_dict= {}
    if locname:
        address= locname.address
        # get the state and the country 
        splited =  address.split(',')
        country = splited[-1].strip()
        state = splited[-2].strip()
        if state.isdigit():
            state = splited[-3].strip()

        address_dict= {'address': address, 'country': country, 'state': state, 'lat' : lat_lon[0], 'lon' : lat_lon[1] }
    return address_dict
    

def get_localizations_from_directory(directory):
    # Get a list of all .tif files in the directory
    image_files = glob.glob(os.path.join(directory, '*.tif'))
    
    localizations = {}
    stations= []
    for image_file in image_files:
        # Extract the filename from the full path
        image_name = os.path.basename(image_file)
        # Extract the localization from the filename
        localization = extract_station(image_name)
        
        if localization:
            stations.append(localization)
            localizations[image_name]  = localization
        else:
            localizations[image_name] = 'No localization found'
    
    return localizations, list(set(stations))


# %%

def get_all_exact_localisations(directory):
    _, stations = get_localizations_from_directory(directory)
    addresses_dict ={}
    for station in stations: 
        lat_long  = get_lat_long_from_station(station)
        address_dict = get_state_country(lat_long)
        addresses_dict[station] = address_dict
    return addresses_dict


def get_stations_all_infos(stations_address, image_to_station): 
    # Initialize a new dictionary to group by station name
    station_to_images = {}

    # Iterate over the original dictionary
    for image, station in image_to_station.items():
        # Initialize the station key if not already present
        if station not in station_to_images:
            station_to_images[station] = {'images': []}
        
        # Add the image to the corresponding station group
        station_to_images[station]['images'].append(image)
    for station in station_to_images.keys():
         station_to_images[station]['count_images'] = len(station_to_images[station]['images'])
    return  merge_dicts(station_to_images, stations_address)



def get_stations_all_infos_directory(directory):
    """ 
    Input: path of the images (input_tiles)
    Main function that extract the information of the stations from the images then 
    for each station  get these infos => ['images', 'count_images', 'address', 'country', 'state', 'lat_lon']
    """
    addresses_dict= get_all_exact_localisations(directory)
    images_station, stations = get_localizations_from_directory(directory)
    stations_info = get_stations_all_infos( addresses_dict, images_station)

    return stations_info

stations_info = get_stations_all_infos_directory(directory)


## Stats stations 


def stations_by_country(stations_info_dict, print_stats= False): 
    # Initialize a new dictionary to group by country
    grouped_by_country = {}

    # Iterate over the original dictionary
    for station, info in stations_info_dict.items():
        # Skip entries with no information
        if not info:
            continue
        
        country = info.get('country', 'Unknown')
        
        # Initialize the country key if not already present
        if country not in grouped_by_country:
            grouped_by_country[country] = []
        
        # Add the station to the corresponding country group
        grouped_by_country[country].append(station)

    # Print the result
    if print_stats : 
        for country, stations in grouped_by_country.items():
            print(f"Country: {country}")
            print(f" number of  Stations: {len(stations)}")
    
    return grouped_by_country


## Stats Images 

def images_by_country(stations_info_dict, print_stats= False): 
    # Initialize a new dictionary to group by country
    grouped_by_country = {}

    # Iterate over the original dictionary
    for station, info in stations_info_dict.items():
        # Skip entries with no information
        if not info:
            continue
        
        country = info.get('country', 'Unknown')
        
        # Initialize the country key if not already present
        if country not in grouped_by_country:
            grouped_by_country[country] = []
        
        # Add the station to the corresponding country group
        grouped_by_country[country].extend(stations_info_dict[station]['images'])

    # Print the result
    if print_stats:
        for country, images in grouped_by_country.items():
            print(f"Country: {country}")
            print(f" number of  Images: {len(images)}")
    
    return grouped_by_country


def images_by_country_and_state(stations_info_dict, print_results= False): 
    # Initialize a new dictionary to group by country
    grouped_by_country_state = {}

    # Iterate over the original dictionary
    for station, info in stations_info_dict.items():
        # Skip entries with no information
        if not info:
            continue
        
        country = info.get('country', 'Unknown')
        state = info.get('state', 'Unknown')
        
        # Initialize the country key if not already present
        if country not in grouped_by_country_state:
            grouped_by_country_state[country] = {}
        
        # Initialize the state key if not already present under the country
        if state not in grouped_by_country_state[country]:
            grouped_by_country_state[country][state] = []
        
        # Add the station to the corresponding country and state group
        grouped_by_country_state[country][state].extend(stations_info_dict[station]['images'])

    if print_results :
        # Print the result
        for country, states in grouped_by_country_state.items():
            print(f"Country: {country}")
            total_country= 0
            for state, images in states.items():
                print(f"    State: {state}")
                print(f"        number of  Images: {len(images)}")
                total_country+= len(images)
            
            print('Total country', total_country )

    
    return grouped_by_country_state


def images_by_country_and_state_and_station(station_details, print_results= False ): 
  
    # Initialize a new dictionary to group by country and state
    grouped_data = {}

    # Iterate over the station details
    for station, details in station_details.items():
        if not details:
            continue
        
        country = details.get('country', 'Unknown')
        state = details.get('state', 'Unknown')
        
        # Initialize country and state keys if not already present
        if country not in grouped_data:
            grouped_data[country] = {}
        
        if state not in grouped_data[country]:
            grouped_data[country][state] = {}
        
        # Add the station and its images to the corresponding country and state group
        grouped_data[country][state][station] = station_details.get(station, {}).get('images', [])

   # Print the result
    if print_results:
        for country, states in grouped_data.items():
            print(f"Country: {country}")
            total_country =0 
            for state, stations in states.items():
                total_state= 0
                print(f"    State: {state}")
                for station, images in stations.items():
                    print(f"        Station: {station}")
                    print(f"            number of  Images: {len(images)}")
                    print(f"            number of  prelevement station: {len(images)/5}")

                    total_state+= len(images)
                print('     Total state:', total_state )
            print('Total country:', total_country )

    return grouped_data

