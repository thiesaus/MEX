
import numpy as np
from copy import deepcopy
from data.dataloader import WORDS_MAPPING

from clip import load

# from utils import VIDEOS, tokenize, expression_conversion


def expression_conversion(expression):
    """expression => expression_new"""
    expression = expression.replace('-', ' ').replace('light color', 'light-color')
    words = expression.split(' ')
    expression_converted = ''
    for word in words:
        if word in WORDS_MAPPING:
            word = WORDS_MAPPING[word]
        expression_converted += f'{word} '
    expression_converted = expression_converted[:-1]
    return expression_converted
def similarity_calibration(TEXT_FEAT_DICT, CLS_DICT, a, b, tau):
    fn = lambda x: a * x + b

    cls_dict = deepcopy(CLS_DICT)
    FEATS = np.array([x['feature'] for x in TEXT_FEAT_DICT['train'].values()])
    PROBS = np.array([x['probability'] for x in TEXT_FEAT_DICT['train'].values()])

    for video, video_value in cls_dict.items():
        for obj_id, obj_value in  video_value.items():
            for frame, frame_value in obj_value.items():
                for exp, exp_value in frame_value.items():
                    exp_new = expression_conversion(exp)
                    feat = np.array(TEXT_FEAT_DICT['test'][exp_new]['feature'])[None, :]
                    sim = (feat @ FEATS.T)[0]
                    sim = (sim - sim.min()) / (sim.max() - sim.min())
                    weight = np.exp(tau * sim) / np.exp(tau * sim).sum()
                    prob = (weight * PROBS).sum()
                    new_exp_value = [
                        x + fn(prob) for x in exp_value
                    ]
                    frame_value[exp] = new_exp_value

    return cls_dict