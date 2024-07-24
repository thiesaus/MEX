from PIL import Image, ImageDraw, ImageFont
import random
import torch
import os
from datetime import datetime
from torch import nn

def SampleFilterFunction(frame, bboxes, sentence):
    '''
    Filter function to filter out the samples that we don't want to visualize
    Args:
        frame: PIL.Image
        bboxes: torch.tensor with Tensor([[x1,y1,x2,y2]])
        sentence: str
    Returns:
        bboxes: torch.tensor with Tensor([[x1,y1,x2,y2]])
    '''
    return bboxes

class Visualizer(nn.Module):
    def __init__(self, save_dir,device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),filter_function=SampleFilterFunction):
        super().__init__()
        self.save_dir=save_dir
        current_datetime = datetime.now()
        date_string = current_datetime.strftime("instance_%Y_%m_%d__%H_%M_%S")
        self.image_dir=os.path.join(save_dir,'images',date_string)
        self.filter_function=filter_function
        self.device = device
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def forward(self, frame:Image, bboxes:torch.Tensor,sentence:str,probs:torch.Tensor):
        '''
        Visualize the image with bounding box
        Args:
            frame: PIL.Image
            bboxes: torch.tensor with Tensor([[x1,y1,x2,y2]])
            sentence: str
        '''
        assert (len(bboxes)==0 or (len(bboxes)>0 and len(bboxes[0]) == 4)), 'Bounding box should be in shape [[x1,y1,x2,y2]]'

     
        font = ImageFont.truetype("segoeui.ttf", 23)
        bboxes=self.filter_function(frame,bboxes,sentence)
        output_frame=frame.copy()
        draw = ImageDraw.Draw(output_frame) 
        for i,bbox in enumerate(bboxes):
            color= ( random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))
            op_color= tuple(255 - component for component in color)
            bbox=bbox.cpu()
            left= bbox[0]
            top= bbox[1]
            right= bbox[2]
            bottom= bbox[3]
            prob = probs[i]
            draw.rectangle((left,top,right,bottom), outline =color, width = 3)
            draw.rectangle([bbox[0],bbox[1]-20,bbox[0]+60 + len("{}".format(i))*3,bbox[1]], fill=color)
            draw.text((bbox[0]+5, bbox[1]-20),"{}: {:2f}".format(i,prob) , fill="red", font=font)
        font_sen = ImageFont.truetype("segoeui.ttf", 30)
        draw.text((15, 15), sentence, font=font_sen, fill="red")
        current_datetime = datetime.now()
        date_string = current_datetime.strftime("picture_%Y_%m_%d__%H_%M_%S")
        output_frame.save(os.path.join(self.image_dir,date_string+'.jpg'),"JPEG")
        print('Image saved at: {}'.format(date_string+'.jpg'))
        return bboxes
