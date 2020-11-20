from predict import Predictor
import numpy as np
import os


# CLASS_NAMES = ("nohat","helmets","mask","nomask",)

def crop_person(img,ids,bboxes):
    """
    Input:
        img : the source imgs
        ids : list. person id
        bboxes : list. person bounding box. XYWH formats
    Return:
        (person_imgs,ids)
    """
    person_imgs = []
    for i,_ in enumerate(ids):
        x1,y1,x2,y2 = bboxes[i]
        # print(x,y,w,h)
        cropped = img[y1:y2,x1:x2]
        person_imgs.append(cropped)
    
    return person_imgs,ids

def parse_detector(predictor,img):
    """
    Input:
        predictor: detector
        img: for detect
    Return:
        score,type,bbox
    """
    outpus = predictor(img)
    predictions = outpus["instances"].to("cpu")
    boxes = (predictions.pred_boxes).tensor.numpy() if predictions.has("pred_boxes") else None
    # scores = predictions.scores if predictions.has("scores") else None
    classes = (predictions.pred_classes).numpy() if predictions.has("pred_classes") else None
    # labels = CLASS_NAMES[classes]

    return boxes,classes


def Detector_(predictor,person_imgs,ids,bboxes,model_two_four):
    """
    Input:
         predictor: object detector
         person_img: list. croped person rois.
         ids: list. person id.
         bboxes: list. XYWH formats.
         model_two_four: true for two classes model false for four classes models.
    Return:
         list. [[id,[type1,type2,...],[box1,box2,...]],[id,type,box],...]
        or
         [id,[type1,type2,...]]
         [id,[[x,y,x,y],[x,y,x,y],..]]
    """
    types = []
    bboxs = []
    outputs = []
    box = []
    # add if bboxes is not None:
    for i,id in enumerate(ids):
        lx1,ly1,lx2,ly2 = bboxes[i]
        boxes,classes = parse_detector(predictor,person_imgs[i])
        # print("classes--------------------",classes)
        # affine transform
        if model_two_four:
            x1,y1,x2,y2 = boxes[0]
            x1 = lx1+x1
            y1 = ly1+y1
            x2 = lx2+x2
            y2 = ly2+y2
         
            box.append([x1,y1,x2,y2])
            types.append(classes)
            bboxs.append(box)
        else:
            if classes>1:
                x1,y1,x2,y2 = boxes[0]
                x1 = lx1+x1
                y1 = ly1+y1
                x2 = lx2+x2
                y2 = ly2+y2
                box.append([x1,y1,x2,y2])
                types.append(classes)
                bboxs.append(box)
        # outputs.append(id)
        # outputs.append(types)
        # outputs.append(bboxs)
    return ids,types,bboxs



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # --------------------------TEST crop_person--------------------------------
    import cv2
    img = '001526.jpg'
    img = cv2.imread(img)
    ids = [0,1]
    bboxes = [[231,253,549,811],[519,33,1101,847]]
    person_imgs,ids = crop_person(img,ids,bboxes)
    # cv2.imshow('rr', person_imgs[1])
    # cv2.waitKey(0)
    # ----------------------------PASS-----------------------------------------------

    # --------------------------TEST parse_detector--------------------------------
    yaml_file = "/home/gaokechen/detectron2-master/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"
    weight_file1 = "model_final_het.pth"
    weight_file2 = "model_final.pth"
    confidence_thr = 0.7
    num_classes1 = 2
    num_classes2 = 5
    predictor = Predictor(yaml_file,weight_file2,confidence_thr,num_classes2)
    boxes,classes = parse_detector(predictor,person_imgs[0])

    predictor4 = Predictor(yaml_file,weight_file1,confidence_thr,num_classes1)
    boxes4,classes4 = parse_detector(predictor4,person_imgs[0])
    # print(boxes4)
    # print(classes4)
    # ----------------------------- PASS ---------------------------------------------

    # ----------------------------TEST Detector_--------------------------------------
    ids,types,boxes = Detector_(predictor,person_imgs,ids,bboxes,1)
    print(ids)
    print(types)
    print(boxes)

    # ----------------------------PASS--------------------------------------
