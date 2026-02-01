build
```sh
# bash docker/ultraliricsfork.sh 
bash docker/onstartbuild.sh 
```

download ckpts
```sh
bash downloader/yoloe_pack.sh /mnt/nvme/huggingface
```

download nu mini from official website and just unpack it:  

```sh
/mnt/nvme/rowdata/nu
├── LICENSE
├── maps
├── samples
├── sweeps
└── v1.0-mini
```

detection with seg masks by text promt, single image

```
python3 test/text_promt/run.py
```



