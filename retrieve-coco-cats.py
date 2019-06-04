from pycocotools.coco import COCO
import requests

coco = COCO('/home/federico/Desktop/Deep learning/cat-generator/cocoapi-master/annotations/instances_val2014.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))


# display COCO categories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['cat'])
imgIds = coco.getImgIds(catIds=catIds )
images = coco.loadImgs(imgIds)
print("\nimgIds: ", imgIds)
print("\nimages: ", images)

for im in images:
    print("\nim: ", im)
    img_data = requests.get(im['coco_url']).content
    with open('/home/federico/Desktop/Deep learning/cat-generator/cocoapi-master/downloaded_images2/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)