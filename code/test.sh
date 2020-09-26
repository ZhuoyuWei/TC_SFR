pip3 install pip --upgrade
pip3 install DateTime
pip3 install requests
pip3 install numpy
pip3 install pandas==1.1.2
pip3 install lightgbm
pip3 install category-encoders
pip3 install pickle-mixin
pip3 install tqdm
pip3 install torch torchvision

python3 predate_and_predict.py $1 $2
