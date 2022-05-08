# HybridSN
This is a Keras implementation of paper:*HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification*
## Test result:
### IN dataset
patch size = 25, bands after PCA = 30, train:validation:test = 0.2:0.1:0.7  

**Test loss**:0.049681294709444046  

**Test acc**:98.98954629898071%

**Classification result:**  

                              precision    recall  f1-score   support 

                     Alfalfa       1.00      0.96      0.98        26
                 Corn-notill       1.00      0.99      1.00       800
                Corn-mintill       0.99      0.97      0.98       465
                        Corn       0.99      1.00      0.99       133
               Grass-pasture       0.95      0.99      0.97       270
                 Grass-trees       0.98      0.99      0.98       409
         Grass-pasture-mowed       1.00      0.80      0.89        15
               Hay-windrowed       1.00      1.00      1.00       267
                        Oats       1.00      0.82      0.90        11
              Soybean-notill       0.99      0.99      0.99       545
             Soybean-mintill       1.00      0.99      0.99      1375
               Soybean-clean       0.97      1.00      0.99       333
                       Wheat       0.98      0.99      0.99       115
                       Woods       0.99      1.00      0.99       708
    Building-Gras-Tree-Drive       1.00      0.99      1.00       216
          Stone-Steel-Towers       1.00      0.98      0.99        52

                    accuracy                           0.99      5740
                   macro avg       0.99      0.97      0.98      5740
                weighted avg       0.99      0.99      0.99      5740
                
**Confusion matrix:**
![IN_CM](https://github.com/lzp-cumtb/HybridSN/blob/main/pics/confusion_mat_without_norm1.png)

**Prediction Map:**
![IN_PM](https://github.com/lzp-cumtb/HybridSN/blob/main/pics/pred_map1.jpg)
### SA dataset:
patch size = 25, bands after PCA = 15, train:validation:test = 0.2:0.1:0.7

**Test loss**:0.00021145949722267687  

**Test acc**:99.99340176582336%

**Classification result:**  

                           precision    recall  f1-score   support

       Broc green weeds 1       1.00      1.00      1.00      1125
      Broc green weeds 22       1.00      1.00      1.00      2087
                   Fallow       1.00      1.00      1.00      1107
        Fallow rough plow       1.00      1.00      1.00       781
            Fallow smooth       1.00      1.00      1.00      1499
                  Stubble       1.00      1.00      1.00      2217
                   Celery       1.00      1.00      1.00      2004
         Grapes untrained       1.00      1.00      1.00      6312
     Soy vineyard develop       1.00      1.00      1.00      3474
     Corn sen green weeds       1.00      1.00      1.00      1835
      Lettuce romaine 4wk       1.00      1.00      1.00       598
      Lettuce romaine 5wk       1.00      1.00      1.00      1079
      Lettuce romaine 6wk       1.00      1.00      1.00       513
      Lettuce romaine 7wk       1.00      1.00      1.00       599
       Vineyard untrained       1.00      1.00      1.00      4071
      Vyard verti trellis       1.00      1.00      1.00      1012

                 accuracy                           1.00     30313
                macro avg       1.00      1.00      1.00     30313
             weighted avg       1.00      1.00      1.00     30313

**Confusion matrix:**
![SA_CM](https://github.com/lzp-cumtb/HybridSN/blob/main/pics/confusion_mat_without_norm.png)
**Prediction map:**
![SA_PM](https://github.com/lzp-cumtb/HybridSN/blob/main/pics/pred_map.jpg)
