import pickle
import os 

countries  =['Algérie ⵍⵣⵣⴰⵢⴻⵔ الجزائر', 'Türkmenistan', 'United States']

percentages_train = { 'Algérie ⵍⵣⵣⴰⵢⴻⵔ الجزائر' : 
                    {
                        'Ouargla' : 30, 
                        'Ghardaïa' : 30, 
                        'Illizi' : 30, 
                        'Adrar' : 10 
                        }, 
                    'Türkmenistan' : {'Balkan welaýaty' : 40,
                                    'Lebap welaýaty' : 30, 
                                    'Mary welaýaty' : 30},
                    'United States' : {'Texas': 75, 
                                    'Colorado': 15, 
                                    'Georgia': 5, 
                                    'Kansas': 5}
                    }

percentages_train_smaller = { 'Algérie ⵍⵣⵣⴰⵢⴻⵔ الجزائر' : 
                    {
                        'Ouargla' : 20, 
                        'Ghardaïa' : 20, 
                        'Illizi' : 20, 
                        'Adrar' : 10 
                        }, 
                    'Türkmenistan' : {'Balkan welaýaty' : 30,
                                    'Lebap welaýaty' : 20, 
                                    'Mary welaýaty' : 20},
                    'United States' : {'Texas': 35, 
                                    'Colorado': 15, 
                                    'Georgia': 5, 
                                    'Kansas': 5}
                    }

def subdataset_train(path = './subdata/subdataset_imagenames_smaller.pkl', percentages_train = percentages_train_smaller ): 

    if os.path.exists(path):
        # Load the array from the file
        with open(path, 'rb') as file:
            dataset_images = pickle.load(file)
    else:
        from dataset_stats import stations_info,images_by_country_and_state
        infos_images = images_by_country_and_state(stations_info)
        infos_images  = {key: value for key,value in infos_images.items() if key in countries }
        
        dataset_images = []
        for country in infos_images.keys():
            for state in infos_images[country].keys(): 
                percentage= percentages_train[country][state]
                dataset_images.extend(infos_images[country][state][:percentage])
        # Save the array to a file
        with open(path, 'wb') as file:
            pickle.dump(dataset_images, file)

    return dataset_images 
dataset_images_train  = subdataset_train()
