import os
import json
import numpy as np

# Set the path to your folder where images are stored
folder_path = 'data/person_1_single'

def generate_intrinsics():
    return [
        -2223.21152,
        2422.76352,
        0.502588,
        0.48830700000000005
    ]

# Generate the JSON structure
data = {
    "camera_angle_x": -0.22928764106469704,
    "frames": [],
    "intrinsics": generate_intrinsics()
}

# Function to generate random bbox and expression values
def generate_random_bbox():
    return [
                0.03125,
                0.998046875,
                0.1015625,
                0.818359375
            ]

def generate_random_expression():
    return [
                -1.16294e-05,
                0.133139,
                -0.124174,
                0.0108487,
                -0.135844,
                0.0283159,
                -0.0466046,
                -0.138834,
                -0.243354,
                0.039138,
                -0.0826465,
                0.368024,
                -0.128166,
                0.032144,
                0.33342,
                0.0553275,
                0.0334842,
                0.170309,
                0.206483,
                -0.0206751,
                -0.0796748,
                0.0882853,
                -0.232236,
                0.120364,
                0.195657,
                0.0865242,
                0.0678549,
                -0.0755545,
                -0.0756427,
                0.103254,
                0.303474,
                -0.0128664,
                0.113038,
                -0.0356958,
                -0.194319,
                -0.0803969,
                -0.037366,
                0.0661746,
                0.0991709,
                0.170522,
                -0.0745419,
                0.441857,
                -0.186441,
                -0.0393437,
                -0.240151,
                0.16473,
                -0.0254639,
                -0.0590731,
                0.155358,
                0.052642,
                -0.187791,
                -0.148488,
                0.033676,
                -0.0219588,
                0.0371149,
                0.0181471,
                -0.108117,
                -0.0523899,
                -0.00987982,
                -0.0374567,
                -0.0726853,
                0.0227449,
                0.240043,
                0.137705,
                0.0170319,
                -0.195727,
                -0.164161,
                -0.00630608,
                -0.229416,
                0.261489,
                0.299902,
                0.092594,
                0.0263351,
                0.167496,
                0.055752,
                0.0149392
            ]

def generate_random_transform_matrix():
    return [
                [
                    0.975096,
                    -0.0213444,
                    -0.220754,
                    -0.12505983864469886
                ],
                [
                    0.0591482,
                    0.984335,
                    0.166091,
                    0.09199261821349274
                ],
                [
                    0.213751,
                    -0.175011,
                    0.961083,
                    0.5170857417970655
                ],
                [
                    -0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]


# Create 100 frames
for i in range(100):
    frame = {
        "bbox": generate_random_bbox(),
        "expression": generate_random_expression(),
        "file_path": f"./train/f_{i:04d}",
        "transform_matrix": generate_random_transform_matrix()
    }
    data["frames"].append(frame)

# Define the output JSON file path
output_json_path = os.path.join(folder_path, 'transforms_train.json')

# Save the JSON to a file
with open(output_json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file has been created at {output_json_path}")
