pip3 install pip --upgrade
pip3 install DateTime
pip3 install requests
pip3 install numpy
pip3 install pandas==1.1.2
pip3 install lightgbm
pip3 install category-encoders
pip3 install pickle-mixin
pip3 install tqdm
pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

python3 predate_and_predict.py $1 $2
