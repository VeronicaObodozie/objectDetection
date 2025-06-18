# 1. Base Template structure for object detection

Benchmark detection tools and frameworks, decided to use detectron2 toolbox. Performance, training/inference speed is slightly better than MMDet. MMDet is also not being supported, hence some compatability issues come up. Pytorch pretrained models were initially used in base code when learning, but detectron2 provides a more customizable pipeline. MMDet does offer more models though


Start with Visdrone and AI-TOD
## Environment setup Requirements
pip install ultralytics
detectron2
numpy, pandas, opencv, torch, scipy, pillow

# 2. Object Detection directory
Object detection repository


## Interesting Datasets

### Helpful dataset lists
* https://github.com/Seyed-Ali-Ahmadi/Awesome_Satellite_Benchmark_Datasets
* https://github.com/coderonion/awesome-object-detection-datasets
* 

### Simple Starting Point
- [Aerial Cars](https://github.com/jekhor/aerial-cars-dataset) : Dataset of drone images. Text

### Environmental based
- [Marine Debris](https://cmr.earthdata.nasa.gov/search/concepts/C2781412735-MLHUB.html)
- [AU-AIR Dataset](https://bozcani.github.io/auairdataset) : Traffic detection using UAVs at low altitudes.
- https://github.com/AICyberTeam/2020Gaofen

### Small Object Datasets
- [AI TOD](https://github.com/jwwangchn/AI-TOD): This is a tiny object detection dataset
- [SODA](https://shaunyuan22.github.io/SODA/) : This is an open source large scale dataset. A has 2513 high-resolution images of aerial scenes, which has 872069 instances annotated with oriented rectangle box annotations over 9 classes. Major bias [clear urban picture in Africa noted as rural.
- [VisDrone](https://github.com/VisDrone/DroneVehicle) : Large scale drone dataset

- [ImageNet](https://image-net.org/challenges/LSVRC/) : "ImageNet Large Scale Visual Recognition Challenge".
- [LEVIR Ship detection](https://github.com/WindVChen/LEVIR-Ship) : Tiny ship detection

- [Multi scene dataset](https://github.com/Hua-YS/Multi-Scene-Recognition) muliscene recognition