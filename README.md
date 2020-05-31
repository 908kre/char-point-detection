```
kaggle competitions download global-wheat-detection -p /store
cd /store
unzip global-wheat-detection.zip && rm global-wheat-detection.zip
mkdir images
mv train/* images
mv test/* images
```
