# Cat-faces-dataset
Dataset containing around 7000 images of cats' faces of size 64x64.

The images were retrieved from 4 different open datasets, namely:
* Cats and Dogs Breeds Classification Oxford Dataset (https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)
* Cute Cats and Dogs from Pixabay.com (https://www.kaggle.com/ppleskov/cute-cats-and-dogs-from-pixabaycom)
* Cat Dataset (https://www.kaggle.com/crawford/cat-dataset)
* COCO dataset (http://cocodataset.org/#home)

The images were preprocessed with an openCV cat face detector (https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/), so that each image would contain only the face of the cats. After that, the false positive cats were removed by a visual inspection
