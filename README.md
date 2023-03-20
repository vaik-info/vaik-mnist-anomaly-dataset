# vaik-mnist-anomaly-dataset

Create MNIST anomaly detection dataset

## Example

![mnist-anomaly-dataset](https://user-images.githubusercontent.com/116471878/226257754-63bb05af-a691-4c63-98d3-b45a4ad527fa.png)


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

![mnist-anomaly-dataset](https://user-images.githubusercontent.com/116471878/226257754-63bb05af-a691-4c63-98d3-b45a4ad527fa.png)
