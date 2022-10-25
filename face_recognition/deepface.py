import numpy as np
import cv2
from face_recognition import face_align
from face_recognition.backbones import get_model
from numpy.linalg import norm as l2norm
import torch


class _Feature:
    def __init__(self, optionalRelease, mandatoryRelease, compiler_flag):
        self.optional = optionalRelease
        self.mandatory = mandatoryRelease
        self.compiler_flag = compiler_flag

    def getOptionalRelease(self):
        """Return first release in which this feature was recognized.

        This is a 5-tuple, of the same form as sys.version_info.
        """

        return self.optional

    def getMandatoryRelease(self):
        """Return release in which this feature will become mandatory.

        This is a 5-tuple, of the same form as sys.version_info, or, if
        the feature was dropped, is None.
        """

        return self.mandatory

    def __repr__(self):
        return "_Feature" + repr((self.optional,
                                  self.mandatory,
                                  self.compiler_flag))


CO_FUTURE_DIVISION = 0x2000   # division
division = _Feature((2, 2, 0, "alpha", 2),
                    (3, 0, 0, "alpha", 0),
                    CO_FUTURE_DIVISION)


class ArcFacePyTorch:
    def __init__(self, model_file=None, net=None, device='cpu'):
        assert model_file is not None
        assert net is not None
        self.model_file = model_file
        self.net = net
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Model face recognition: {}, device: {}".format(model_file.split('/')[-1], self.device))
        self.model = get_model(self.net, num_features=512, fp16=False)
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval().to(device=self.device)

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps)
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat, feat_data):
        from numpy.linalg import norm
        sim = np.sum(feat_data * feat, axis=1) / (np.sqrt(np.sum(feat_data ** 2, axis=1)) * norm(feat))
        sim_max = max(sim)
        idx = np.argmax(sim)
        return sim_max, idx

    # def compute_sim(self, feet1, feet2):
    #     from numpy.linalg import norm
    #     feet11 = feet1.ravel()
    #     feet21 = feet2.ravel()
    #     sim = np.dot(feet11, feet21) / (norm(feet11) * norm(feet21))
    #     return sim

    def get_feat(self, imgs):
        img = cv2.resize(imgs, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.to(device=self.device)
        torch.cuda.empty_cache()
        import time
        feet = self.model(img).cpu().detach().numpy()
        return feet.ravel()


class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'




