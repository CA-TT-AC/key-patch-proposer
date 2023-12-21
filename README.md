# Key Patch Proposer(KPP) algorithm
This repo is a modification on the [MAE repo](https://github.com/facebookresearch/mae).
## Environment
You should follow the instruction of MAE to prepare your environment. https://github.com/facebookresearch/mae

## Dateset
To prepare images to be applied to KPP algorithm, you should collect them into a folder or you can rewrite dataset class in ```custom_datasets.py```.

## Run
Run the following:
```
python KPP_main.py --data_path /path/to/your/images --ckpt /your/pretrained/ckpt.pth --output_dir /your/ourpur/dir --mask_ratio 0.75
```
You can also use ```--visulize True```, to visulize what is generated by KPP algorithm in your images.