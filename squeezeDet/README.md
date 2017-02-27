## _SqueezeDet

This is a modified version of the original SqueezeDet library, compatible with Tensorflow 1.0.
    
## Installation:
- Prerequisites:
    - Follow instructions to install Tensorflow: https://www.tensorflow.org. Version: 0.11.0rc0
    - Install opencv: http://opencv.org
    - Other packages that you might also need: easydict, joblib. You can use pip to install these packages:
    
    ```Shell
    pip install easydict
    pip install joblib
    ```
- Clone the SqueezeDet repository:

  ```Shell
  git clone https://github.com/BichenWuUCB/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet as `$SQDT_ROOT`. 

## Demo:
- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `$SQDT_ROOT/data/` If you are using command line, type:

  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
  tar -xzvf model_checkpoints.tgz
  rm model_checkpoints.tgz
  ```


- To detect an image in `$SQDT_ROOT/data/imag.png`,

  ```Shell
  cd $SQDT_ROOT/
  python ./src/classify.py --input_path=./data/imag.png
  ```
