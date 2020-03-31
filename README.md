https://www.kaggle.com/c/imet-2020-fgvc7

```sh
docker-compose run --rm app bash
```

dataset setup
```sh
cd /store
kaggle competitions download -c imet-2020-fgvc7 -p dataset
```
# idea

* アスペクト比を考慮する
* カテゴリ別にモデルを作成する

# IMET-2019

solutions
## 1th: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687
## 2th: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/95223
## 3th: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/96149
## 4th: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94817
### Augmentations
* Random crop on train and center crop on validation and test


## 10th: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687

# fixed

* pseudo lableling

