# vaik-mnist-anomaly-dataset

Create MNIST anomaly detection dataset

## Example

![vaik-mnist-anomaly-dataset](https://user-images.githubusercontent.com/116471878/225503307-4c28b9e2-981c-4413-8cad-0b7d77f31231.png)


## Usage

```shell
pip install -r requirements.txt
python main.py --output_dir_path ~/.vaik-mnist-anomaly-dataset \
                --train_sample_num 12500 \
                --train_anomaly_ratio 0.0 \
                --valid_sample_num 100 \
                --valid_anomaly_ratio 0.8 \
                --image_max_size 256 \
                --image_min_size 196
```

## Output

```shell
~/.vaik-mnist-anomaly-dataset$ tree
.
├── train
│   └── good
│       ├── 0000_23269.png
│       ├── 0001_8229.png
│       ├── 0002_22674.png
│       ├── 0003_11270.png
│       ├── 0004_3029.png
│       ├── 0005_35791.png
│       ├── 0006_28364.png
│       ├── 0007_19185.png
│       └── 0008_4225.png
└── valid
    └── anomaly
        ├── ground_truth
        │   ├── line
        │   │   ├── 0002_9407.png
        │   │   ├── 0003_9883.png
        │   │   └── 0006_2234.png
        │   ├── mix
        │   │   └── 0000_162.png
        │   └── point
        │       ├── 0001_4572.png
        │       ├── 0004_3711.png
        │       ├── 0005_1341.png
        │       └── 0007_6785.png
        └── test
            ├── good
            │   └── 0000_6805.png
            ├── line
            │   ├── 0002_9407.png
            │   ├── 0003_9883.png
            │   └── 0006_2234.png
            ├── mix
            │   └── 0000_162.png
            └── point
                ├── 0001_4572.png
                ├── 0004_3711.png
                ├── 0005_1341.png
                └── 0007_6785.png

```

- good image, anomaly image, and ground truth image

![vaik-mnist-anomaly-dataset](https://user-images.githubusercontent.com/116471878/225503307-4c28b9e2-981c-4413-8cad-0b7d77f31231.png)
