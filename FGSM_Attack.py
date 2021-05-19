from os import system
import torch
import sys
# system('pip install git+https://github.com/ruotianluo/ImageCaptioning.pytorch.git')
# system('pip install gdown')
# system('gdown --id 1VmUzgu0qlmCMqM1ajoOZxOXP3hiC_qlL')
# system('gdown --id 1zQe00W02veVYq-hdq5WsPOS3OPkNdq79')
# system('pip install yacs')
# system('git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git')
# sys.path.append('/vqa-maskrcnn-benchmark')
# system('python setup.py build')
# system('python setup.py develop')
# system('wget -O detectron_model.pth wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth')
# system('wget -O detectron_model.yaml wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml')



import captioning
import captioning.utils.misc
import captioning.models
import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

infos = captioning.utils.misc.pickle_load(open('infos_trans12-best.pkl', 'rb'))
infos['opt'].vocab = infos['vocab']

model = captioning.models.setup(infos['opt'])
model.to('cuda')
model.load_state_dict(torch.load('model-best.pth'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode(image, caption, vocab, device): #image in .jpg, caption - list of words (tokens)
    voc = {value : key for (key, value) in vocab}
    labels = torch.IntTensor([int(int(voc.get(w, 0)) + 1) for w in caption]).reshape(1, 1, -1).to(device)
    nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, labels)))
    mask_batch = np.zeros([labels.shape[0], len(caption) + 2], dtype = 'float32')
    for ix, row in enumerate(mask_batch):
        row[:nonzeros[ix]] = 1
    mask = torch.Tensor(mask_batch.reshape(1, 1, -1)).to(device)
    
    im = np.array(image).astype(np.float32)
    im = im[:, :, ::-1]
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1333:
          im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
          im,
          None,
          None,
          fx=im_scale,
          fy=im_scale,
          interpolation=cv2.INTER_LINEAR
      )
    img = torch.from_numpy(im).permute(2, 0, 1)
    img_tensor, im_scales = [img], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32).to(device)
    return current_img_list, labels, mask, im_scales

from torch import nn

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)
        target = target.long()
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)
        return output
    
import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd


import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


class FeatureExtractor:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  
  def __init__(self):
    self.detection_model = self._build_detection_model()
  
  def _build_detection_model(self):

      cfg.merge_from_file('/content/model_data/detectron_model.yaml')
      cfg.freeze()

      model = build_detection_model(cfg)
      checkpoint = torch.load('/content/model_data/detectron_model.pth', 
                              map_location=torch.device("cpu"))

      load_state_dict(model, checkpoint.pop("model"))

      model.to("cuda")
      model.eval()
      return model

  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list

def fgsm_attack(image, epsilon, data_grad): # perturbed_image = image + epsilon * grad
    sign_data_grad = data_grad.sign()
    perturbed_image = image
    perturbed_image.tensors = perturbed_image.tensors + epsilon*sign_data_grad
    perturbed_image.tensors = torch.clamp(perturbed_image.tensors, 0, 1)
    return perturbed_image
    
def main():
    infos = captioning.utils.misc.pickle_load(open('infos_trans12-best.pkl', 'rb'))
    infos['opt'].vocab = infos['vocab']

    model = captioning.models.setup(infos['opt'])
    model.to('cuda')
    model.load_state_dict(torch.load('model-best.pth'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = FeatureExtractor()
    print('Введите путь к интересующему Вас изображению')
    image = Image.open(input()) #load your image
    print('Введите корректную подпись к фото')
    caption = input().split(' ') #insert correct caption for your image
    current_img_list, labels, mask, im_scales = encode(image, caption, infos['opt'].vocab .items(), device)
    current_img_list.tensors.requires_grad = True
    output = feature_extractor.detection_model(current_img_list)
    img_features = feature_extractor._process_feature_extraction(output, im_scales, 
                                                'fc6', 0.2)[0]
    model.eval()
    crit = LanguageModelCriterion()
    loss = crit(model(img_features.mean(0).reshape(1, -1), img_features.reshape(1, 100, 2048), labels[:, :, :-1]), labels[:, :, 1:], mask[:, :, 1:])
    model.zero_grad()
    feature_extractor.detection_model.zero_grad()
    loss.backward()
    print('Введите параметр для FGSM атаки - Epsilon')
    epsilon = float(input()) # input epsilon > 0
    perturbed_image = fgsm_attack(current_img_list, epsilon, current_img_list.tensors.grad.data)
    output = feature_extractor.detection_model(current_img_list)
    img_features = feature_extractor._process_feature_extraction(output, im_scales, 
                                                'fc6', 0.2)[0]
    perturbed_caption = model.decode_sequence(model(img_features.mean(0)[None], img_features[None], mode='sample', opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':1})[0])
    print(perturbed_caption)
    
if __name__ == "__main__":
    main()
