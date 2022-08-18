## Ego-Lane Detection with Vector Fields 
This repository has been designed for the Carolo-Cup context.
The goal was to **predict the left marking of the ego-lane**
which usually corresponds to the center marking of the road.
Since the Carolo-Cup provides a strictly defined competition environment,
it is always clear where to go from any image, so that the network
has been set up to learn any turns from context.

The image undergoes a perspective transform (IPM) prior to the neural network,
so that it looks like a bird's eye view.
**The position of the car always corresponds to the bottom center of the image**.
In the example image below, the predicted vector field
is visualized by the red arrows.
![img_1.png](doc/img_1.png)

### Idea
Generally, the idea is to let the network predict a vector field 
with an attractor basin in order to describe lane markings.
The advantage of this vector field form of representation is
that it aligns well with the spatial data structure of the images -- this 
simplifies the task to be learned.
For further work, this project could also be interesting for predicting 3D trajectories
from point clouds or when depth information is available.

### Architecture
Since useful semantics are expected to be learned by a semantic segmentation, 
the network BiSeNetV2 has been chosen as a **backbone** for this lane detection
network.
BiSeNetV2 has been chosen over alternatives due to its low inference time.

The main output is the vector field (VF) that represents the basin for the
center marking of the road.
The actual vector field results from a linear interpolation of a tensor.
As it can be seen from the examples, a low resolution of 12x20 can still produce
detailed vector fields thanks to the interpolation. This allows to work 
with fairly low resolutions and thus highly contextual layers.

A second vector field (DF) is used to predict the driving direction.
A third output is used to grade the visibility conditions on the image
(yet to be improved).

Input image resolution: 320x192.
![img_2.png](doc/img_2.png)


### Vector Field Parsing
#### (Extraction of the lane marking geometry from the vector field)
This is done using iterative vector field integration with momentum.
Since the car position corresponds to the bottom center,
the integration can start from there. 
In practice, it is advisable to perform the integration on the CPU
due to its iterative nature.

The integration algorithm works by alternating 
_integration steps_ (white) and _correction steps_ (magenta).
The green line is the final lane marking shape.
![img_3.png](doc/img_3.png)


### Loss
Due to its iterative and somewhat recursive nature, it is impractical to use
the ground truth lane marking curves directly as labels.
Furthermore, it is sub-optimal to generate references for the vector field
tensor, as there is not necessarily a correct value for the cells
far from the lane.

So, the idea is to train the network exactly for what it should do in the end:
**Train for correction steps** of the integration algorithm.

The ground truth is available in the form of an equidistantly sampled line.
1. Starting from the ground truth (blue), perturb points randomly.
2. Let the network perform correction steps (green).
3. Ideally, the corrections should have landed on the blue line. Everything else
is an error that should contribute to the loss function (so: red lines are errors).

![img_4.png](doc/img_4.png)


### How to use the code (English translation is to be done)

### Workflow

1. Training der Segmentierung
    - Learning Rate Range Test nicht vergessen: Alle anderen Hyper-Parameter einstellen, dann als `--lr_scheduler lr_range_test` angeben. 
    Dauert ein paar Minuten.
    Dann das Diagramm `lr_range_test.svg` im angegebenen log Ordner anschauen (x: Learning Rate, y: Loss) und überlegen, 
    welche LR maximal gewählt werden kann, damit das Training noch konvergiert (möglichst ein Wert kurz vor dem Minimum im Diagramm).
    - Danach `--lr_scheduler 1cycle` oder `--lr_scheduler warmup_poly` wieder einstellen 
2. Training der Spurerkennung mit der segmentation checkpoint als Initialisierung (wird gefreezed)
    - LR range test auch hier nicht vergessen
3. `trace_model.py` zum exportieren für torch::jit. Dafür beide checkpoints angeben.
 Dafür braucht man ein kombiniertes NN mit beiden outputs als Tuple, wenn noch nicht geschehen 
(Klasse anlegen und beide Netze zusammen mergen. Dabei ist wichtig, dass die Namen der Submodules stimmen, damit die weights geladen werden können). 
Wie es zum Beispiel bei `"bisenetv2+lane"` der Fall ist.

### Monitoring
Im angegebenen log Ordner für das Training werden für jede Verbesserung auf den Validierungsdaten Bilder
abgelegt mit den Outputs, die das Netz auf dem entsprechenden Stand ausgibt.
Das Format ist so, dass man mit dem visualize_labels.py aus dem data generation repository visualisieren kann.
Außerdem wird die Datei `train_statistics.json` im log Ordner nach jeder Epoche geupdatet, woraus mit dem Code im `diagrams/` Ordner Diagramme erzeugt werden können.

### Aufbau (ein paar ausgewählte Aspekte)
- `train.py` entry point für das Training, mit Kommandozeilenparametern, Training-Loop, usw.
- `eval.py` entry point und code für evaluation mit validation oder test daten
- `trace_model.py` Exportieren für die Inference auf dem Auto
- `dataset/` Data Loading, Augmentation
- `model/` Hier sollen alle model definitions für die Neuronalen Netze rein
- `nn_utils/` Utility functions die überall gebraucht werden. Losses, LR scheduler, Metriken, math
- `diagrams/` Funktionen um aus `train_statistics.json` Dateien Diagramme über das Training zu erstellen.
  
### Training Segmentierung: BiSeNetV2
#### Pretrained weights download
Diese weights können als starting point für das training benutzt werden:  
https://drive.google.com/file/d/1qq38u9JT4pp1ubecGLTCHHtqwntH0FCY/view
#### Paper
https://arxiv.org/pdf/2004.02147.pdf
#### Command für Obstacle Mode Training
```
python3.6 train.py bisenetv2 
  --bisenetv2_aux_num 4
  --test_step_exp 1
  --pretrained_model_path ~/Downloads/model_final_v2.pth
  --save_model_path log/obstacle_320x192_crop20
  --loss ohem_ce
  --num_classes 19
  --data /home/isf/prepared_isfl_generated_train_data/crop20/obstacle_mode_2d_19_ipm_camvid_mono/
  --crop_height 192
  --crop_width 320
  --batch_size 16
  --eval_batch_size 32
  --learning_rate 0.05
  --lr_scheduler warmup_poly
  --lr_range_test_it 500
  --num_workers 12
  --ignore_class ignore
  --weight_decay 5e-4
  --optimizer sgd
  --num_epochs 30
  --cuda 0
  --grad_momentum 0.9
```
### Training Spurerkennung
Ungefähr in dieser Richtung:
```
python3.6 train.py proj_xy_no_ce
--data ~/isfl_generated_train_data/obstacle_mode_2d_19_ipm_camvid_mono
--test_step_exp 5
--save_model_path log_lane_detection
--loss lane_integration_sampling lane_main_flow lane_visibility_grid
--crop_width 320
--crop_height 192
--batch_size 16
--learning_rate 0.03
--num_epochs 50
--optimizer sgd
--pretrained_model_path ~/catkin_ws/src/ros/perception/data/segmentation_obstacle.pt
--lr_scheduler warmup_poly
--eval_batch_size 16
--ignore_class ignore
--main_val_metric lane_f1_mean
--augmentation viz_aug=0.1
--num_workers 8
```


## Related / Credits
- https://github.com/CoinCheung/BiSeNet
This is the underlying BiSeNet implementation.