# pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/CLIP
# pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip
# pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/lvis-api
# pip install git+https://github.com/THU-MIG/yoloe.git
# pip install -r third_party/sam2/requirements.txt
# pip install -e third_party/sam2/

PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore python3 -m pip install --upgrade pip
PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install ./yoloefork/third_party/CLIP
PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install ./yoloefork/third_party/ml-mobileclip
PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install ./yoloefork/third_party/lvis-api
PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install ./yoloefork
# PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install -r ./yoloefork/third_party/sam2/requirements.txt --verbose --verbose
# PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 pip install -e ./yoloefork/third_party/sam2/ --trusted-host pypi.tuna.tsinghua.edu.cn --verbose --verbose
