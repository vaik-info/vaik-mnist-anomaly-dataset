# vaik-mnist-anomaly-dataset

Create MNIST anomaly detection dataset

## Example

![vaik-mnist-anomaly-dataset](https://user-images.githubusercontent.com/116471878/225503307-4c28b9e2-981c-4413-8cad-0b7d77f31231.png)


## Usage

```shell
pip install -r requirements.txt
python main.py --output_dir_path ~/.vaik-mnist-anomaly-dataset \
                --target_digit 5 \
                --train_sample_num 10000 \
                --valid_sample_num 100 \
                --image_max_size 232 \
                --image_min_size 216
```

## Output

```shell
~/.vaik-mnist-anomaly-dataset$ tree
.
├── train
│   └── good
│       ├── 00000000.png
│       ├── 00000001.png
│       ├── 00000002.png
│       ├── 00000003.png
│       ├── 00000004.png
│       ├── 00000005.png
│       ├── 00000006.png
│       ├── 00000007.png
│       ├── 00000008.png
│       └── 00000009.png
└── valid
    ├── anomaly
    │   ├── 00000000.png
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   ├── 00000003.png
    │   ├── 00000004.png
    │   ├── 00000005.png
    │   ├── 00000006.png
    │   ├── 00000007.png
    │   ├── 00000008.png
    │   └── 00000009.png
    └── good
        ├── 00000000.png
        ├── 00000001.png
        ├── 00000002.png
        ├── 00000003.png
        ├── 00000004.png
        ├── 00000005.png
        ├── 00000006.png
        ├── 00000007.png
        ├── 00000008.png
        └── 00000009.png
```

- good image, anomaly image
