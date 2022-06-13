# Introduction
A Docker image was chosen as a deployment method as it is very easy to test and replicate on any machine where docker is installed.

This README shows how to build the docker image and run it with a few commands. You can check the source code in `/model/src`.

This docker container boots, reads the files in the input directory, runs the analysis, writes the results in the output folder and then exits.

# Building The Image
```bash
cd task-5-deployment
docker build . -t cypress_model:latest
```

#cd /samba/brainpooluser/orthomosaics/DockerContainer-Cypress/Server-v.0.0/task-5-deployment


# Running The Image
## Input Format
Input directory must contain the following:
```
input
│   NDVI.tif    (tif files pertain to the same date and field)
│   therm.tif   
│   salin.tif
└───holes_outlines
    │   holes_outlines.shp  (contains the geometries of the holes that are within the bounds of the input tif files)
    │   ...
```

## Run Command
```bash
sudo docker run --rm -it \
-v {{absolute/path/to/input/dir}}:/app/input \
-v {{absolute/path/to/output/dir}}:/app/output \
cypress_model python ./main.py --analysis smi thresholding
```
#docker run --rm -it -v /samba/brainpooluser/orthomosaics/DockerContainer-Cypress/Server-v.0.0/task-5-deployment/input:/app/input -v /samba/brainpooluser/orthomosaics/DockerContainer-Cypress/Server-v.0.0/task-5-deployment/output/output:/app/output cypress_model python ./main.py --analysis smi thresholding

## Output Format
```
output/
├── smi
│   └── smi.tif     (SMI Map)
└── thresholding
    ├── Unhealthy_ndvi.gpkg
    ├── Unhealthy_ndvi.tif
    ├── Waterlogged.gpkg
    ├── Waterlogged.tif
    ├── Waterstress.gpkg
    └── Waterstress.tif
└── postproc
    ├── coloredNDVI.tif
    ├── coloredSMI.tif
    ├── coloredSalin.tif
    └── coloredTherm.tif

```
