# Real-time-Age-Prediction

## Dataset

The age is annotated in the file name as "age.#.jpg"
<img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput1.png" height="280"/> <img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput2.png" height="280"/> <img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput1.png" height="280"/> <img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput2.png" height="280"/> 



Create new folder to save your own data
* data
  * train
  * test
  * validate

## Train

Create new folder to save your model
* model
  * SSRNet

Use your dataset to train the model
```sh
python SSRNet_train.py
```

## Test

Test the trained model with your test set
```sh
python SSRNet_test.py
```
