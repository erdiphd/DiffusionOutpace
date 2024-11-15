# Palette: Image-to-Image Diffusion Models

[Paper](https://arxiv.org/pdf/2111.05826.pdf ) |  [Project](https://iterative-refinement.github.io/palette/ )

This is an adaptation of **Palette: Image-to-Image Diffusion Models**, based on [this](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) implementation. The main modifications here are the implementation of custom masks and changing the input resolution.

## Usage
### Environment

There should be no need for additional installs, outpace env should work out of the box, but just in case, these are the requirements for palette.

```python
pip install -r requirements.txt
```

### Training Data
After you prepared own data, you need to modify the corresponding configure file to point to your data. Take the following as an example:

```yaml
"which_dataset": {  // import designated dataset using arguments 
    "name": ["data.dataset", "InpaintDataset"], // import Dataset() class
    "args":{ // arguments to initialize dataset
    	"data_root": "dataset/train.flist", //update to full directory
    	"data_len": -1,
    	"mask_mode": "path_oriented"
    } 
}
```
The .flist paper contains the full path of the training images, an example is provided in dataset/train.flist

### Training/Resume Training

Update the config file as needed for hyperparameter tuning. The provided config file "inpainting_u_maze_64.json" is for training Point-U Maze goal candidate generator.

Run the script:

```python
python run.py -p train -c config/inpainting_u_maze_64.json
```