from Actionsrecognition.ActionsEstLoader import TSSTG
import numpy as np


action_model = TSSTG()


if __name__ == "__main__":
    pts = np.empty((32,17,3))
    imge_shape = (640,640)
    out = action_model.predict(pts, imge_shape)
    # out = action_model.predict(pts, image.shape[:2])

    action = action_model.class_names[out[0].argmax()]
    action_name = '{}: {:.2f}%'.format(action, out[0].max() * 100)

    print(action_name)