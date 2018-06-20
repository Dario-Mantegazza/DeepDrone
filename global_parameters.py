image_height = 60
image_width = 108

output_names = ["x_pred", "y_pred", "z_pred", "yaw_pred"]

# ------ Global Dictionaries ------

pickle_sections = {
    "1": 1,
    "2": 0,
    "3": 3,
    "4": 4,
    "5": 4,
    "6": 3,
    "7": 2,
    "8": 0,
    "9": 1,
    "10": 3,
    "11": 4,
    "12": 2,
    "13": 3,
    "14": 4,
    "15": 1,
    "16": 0,
    "17": 2,
    "18": 2,
    "19": 2,
    "20": 1,
    "21": 0,
    "22": 4
}
bag_end_cut = {
    "1": 3150,
    "2": 7000,
    "3": 390,
    "4": 1850,
    "5": 3840,
    "6": 1650,
    "7": 2145,
    "8": 595,
    "9": 1065,
    "10": 2089,
    "11": 1370,
    "12": 5600,
    "13": 8490,
    "14": 4450,
    "15": 7145,
    "16": 3500,
    "17": 1400,
    "18": 1300,
    "19": 1728,
    "20": 5070,
    "21": 11960,
    "22": 5200
}

bag_start_cut = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 58,
    "8": 63,
    "9": 75,
    "10": 50,
    "11": 0,
    "12": 470,
    "13": 40,
    "14": 50,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 220,
    "21": 0,
    "22": 222
}

bag_file_path = {
    "1": "./bagfiles/train/",
    "2": "./bagfiles/train/",
    "3": "./bagfiles/validation/",
    "4": "./bagfiles/validation/",
    "5": "./bagfiles/train/",
    "6": "./bagfiles/validation/",
    "7": "./bagfiles/train/",
    "8": "./bagfiles/train/",
    "9": "./bagfiles/train/",
    "10": "./bagfiles/train/",
    "11": "./bagfiles/train/",
    "12": "./bagfiles/train/",
    "13": "./bagfiles/train/",
    "14": "./bagfiles/train/",
    "15": "./bagfiles/validation/",
    "16": "./bagfiles/train/",
    "17": "./bagfiles/train/",
    "18": "./bagfiles/train/",
    "19": "./bagfiles/train/",
    "20": "./bagfiles/train/",
    "21": "./bagfiles/train/",
    "22": "./bagfiles/validation/"
}
