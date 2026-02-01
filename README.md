deps setup:  

```
bash docker/build_cu124.sh
bash docker/build.sh
bash docker/run.sh
bash docker/attach.sh
```

build when nvidia can compile code(after container start)
```sh
# bash docker/ultraliricsfork.sh 
bash docker/onstartbuild.sh 
```

you can commit the runtime compilations by  
```sh
bash docker/commit.sh
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



