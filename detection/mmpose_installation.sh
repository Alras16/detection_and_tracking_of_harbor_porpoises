# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv-full
# Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one.
# See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# install mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

mkdir data
ln -s $COCO_ROOT detection/mmpose/data/coco
