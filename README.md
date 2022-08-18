## Lane Detection with Vector Fields based on BiSeNetv2 



## How to train (English TBD)

### Workflow

1. Training der Segmentierung
    - Learning Rate Range Test nicht vergessen: Alle anderen Hyper-Parameter einstellen, dann als `--lr_scheduler lr_range_test` angeben. 
    Dauert ein paar Minuten.
    Dann das Diagramm `lr_range_test.svg` im angegebenen log Ordner bewundern (x: Learning Rate, y: Loss) und überlegen, 
    welche LR maximal gewählt werden kann, damit wir noch konvergieren (möglichst ein Wert kurz vor dem Minimum im Diagramm).
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