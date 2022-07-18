import json
from typing import Dict, List

import os
import sys
import ast
import io
import cv2
import copy
import numpy as np
from PIL import Image
from skimage.transform import resize

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# import data drift and AXAI libraries
from model_lib.src.modzy import Utilities
from model_lib.src.modzy_axai import AXAI


# define data directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'unbalanced_model.pth')

"""
The required output structure for a successful inference run for a models is the following JSON:

{
    "data": {
        "result": <inference-result>,
        "explanation": <explanation-data>,
        "drift": <drift-data>,
    }
}

The `data` key is required and stores a dictionary which represents the output for a specific input. The only top-level 
key within these dictionaries that is required is `result`, however, `explanation` and `drift` are additional keys that
may be included if your particular model supports drift detection or explainability. All three of these keys
(`result`, `explanation`, and `drift`) are required to have a particular format in order to provide platform support.
This format type must be specified in the model.yaml file for the version that you are releasing, and the structure for
this format type must be followed. If no formats are specified, it is possible to define your own custom structure on a
per-model basis.

The required output structure for a failed inference run for a models is the following JSON:

{
    "error_message": <error-message>
}

Here, all error information that you can extract can be loaded into a single string and returned. This could be a JSON
string with a structured error log, or a stack trace dumped to a string.

Specifications:
This section details the currently supported specifications for the "result", "explanation", and "drift" fields of each
successful output JSON. These correspond to specifications selected in the `resultsFormat`, `driftFormat`,
`explanationFormat` of the model.yaml file for the particular version of the model.

* `resultsFormat`:

1A) imageClassification

"result": {
    "classPredictions": [
        {"class": <class-1-label>, "score": <class-1-probability>},
        ...,
        {"class": <class-n-label>, "score": <class-n-probability>}
    ]
}

* `driftFormat`

2A) imageRLE

explanation: {
    "maskRLE": <rle-mask>
}

Here, the <rle-mask> is a fortran ordered run-length encoding.

* `explanationFormat`

3A) ResNet50

drift: {
    {
        "layer1": <layer-data>
        "layer2": <layer-data>
        "layer3": <layer-data>
        "layer4": <layer-data>
    }
}

"""

def rle_encode_mask(mask):
    """run length encode a mask in column-major order"""
    mask = np.array(mask, dtype=np.bool_, copy=False)
    curr = 0
    count = 0
    counts = []
    for x in np.nditer(mask, order="F"):
        if x != curr:
            counts.append(count)
            count = 0
            curr = x
        count += 1
    counts.append(count)
    return counts


def get_success_json_structure(inference_result, explanation_result, drift_result) -> Dict[str, bytes]:
    """Convert inference results, explanation results, and drift results into correct output format"""
    
    output_item_json = {
        "data": {
            "result": inference_result,
            "explanation": explanation_result,
            "drift": drift_result,
        }
    }
    return {"results.json": json.dumps(output_item_json, separators=(",", ":")).encode()}


def get_failure_json_structure(error_message: str) -> Dict[str, bytes]:
    """Format any errors"""
    error_json = {"error_message": error_message}
    return {"error": json.dumps(error_json).encode()}


class GRPCResNetImageClassification:
    # Note: Throwing unhandled exceptions that contain lots of information about the issue is expected and encouraged
    # for models when they encounter any issues or internal errors.

    def __init__(self):
        """
        This constructor should perform all initialization for your model. For example, all one-time tasks such as
        loading your model weights into memory should be performed here.

        This corresponds to the Status remote procedure call.
        """
        # experimenting
        self.utilities = Utilities()
        self.unbalanced_model, self.device = self.utilities.load_weights(WEIGHTS_DIR, None, None)
        self.unbalanced_model, self.arrays, self.hook_handles = self.utilities.install(self.unbalanced_model) 
                
        # labels
        self.labels = ["plane", "empty seafloor", "ship"]
            
        self.pxls = 150
        # define data transform
        self.transform = transforms.Compose([ 
            transforms.Resize(size=(150,150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])        

        # initialize AXAI exp
        self.AXAI = AXAI(self.unbalanced_model)
    
    def preprocess_img_bytes(self,img_bytes):
        """
        Args: 
            input image in bytes format
        Returns: 
            NumPy array of preprocessed image and original shape of input image
        """

        try:
            data = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_shape = data.size
        except Exception as e:
            return get_failure_json_structure(
                f"invalid image: the image file is corrupt or the format is not supported. Exception message: {e}"
            ),None

        # resize and center crop input
        data = self.transform(data).to(self.device)
        data = data.reshape(1,3,self.pxls, self.pxls)
        
        return data, orig_shape
   
    def batch_predict(self,images):
        """
        Args: 
            images: 3-dimensional NumPy array (single image) or 4-dimensional numpy array (N images)
        Returns: 
            softmax_output: NxC (N images, C classes) NumPy array containing softmaxed model output
        """        
        # if single image, add axis
        if len(images.shape)==3:
            images = np.expand_dims(images, axis=0)

        # inference and softmax
        all_formatted_results = []
        output = self.unbalanced_model(images)
        probs = torch.nn.functional.softmax(output, dim=1)
        probs = probs.detach().cpu().numpy()
        for prob in probs:
            prob = np.expand_dims(prob, 0)
            indices = np.argsort(prob[0])[::-1]
            inference_result = {
                "classPredictions": [
                    {"class": self.labels[idx], "score": round(float(prob[0][idx]), 3)}
                for idx in indices]
            }
            all_formatted_results.append(inference_result)            
        
        return all_formatted_results      
    
    def reinstall(self):
        for handle in self.hook_handles:
            handle.remove()
        self.unbalanced_model,self.arrays,self.hook_handles = self.utilities.install(self.unbalanced_model)    
    
    def handle_single_input(self, model_input: Dict[str, bytes], detect_drift: bool, explain: bool) -> Dict[str, bytes]:
        """
        This corresponds to the Run remote procedure call for single inputs.
        """
        # `model_input` will have binary contents for each of the input file types specified in your model.yaml file

        # You are responsible for processing these files in a manner that is specific to your model, and producing
        # inference, drift, and explainability results where appropriate.
        
        result = self.handle_input_batch([model_input], detect_drift, explain)[0]

        return result
  
    def handle_input_batch(self, model_inputs: List[Dict[str, bytes]], detect_drift, explain) -> List[Dict[str, bytes]]:
        """
        This is an optional method that will be attempted to be called when more than one inputs to the model
        are ready to be processed. This enables a user to provide a more efficient means of handling inputs in batch
        that takes advantage of specific properties of their model.

        If you are not implementing custom batch processing, this method should raise a NotImplementedError. If you are
        implementing custom batch processing, then any unhandled exception will be interpreted as a fatal error that
        will result in the entire batch failing. If you would like to allow individual elements of the batch to fail
        without failing the entire batch, then you must handle the exception within this function, and ensure the JSON
        structure for messages with an error has a top level "error" key with a detailed description of the error
        message.

        This corresponds to the Run remote procedure call for batch inputs.

        {
            "error": "your error message here"
        }

        """
        # try to decode image bytes for all input images
        indexed_errors = {}
        imgs = []
        shapes = []
        for i, model_input in enumerate(model_inputs):
            # Try to get a image frame, but otherwise go for the data key
            image = model_input['image']
            load_res, orig_shape = self.preprocess_img_bytes(image)
            if isinstance(load_res,dict):
                indexed_errors[i] = load_res
            else:
                imgs.append(load_res)
                shapes.append(orig_shape)
        
        # if any valid images
        if imgs:
            # concatenate into single array
            X = torch.cat(imgs)

            # run inference
            preds = self.batch_predict(X)

            # format results
            formatted_results_iterator = iter(preds)

            # data drift hooks TODO: fix this
            arrays = self.utilities.process(self.arrays, ".") 
            drift_results = []
            for i in range(len(imgs)):
                drift_result = {"features_1": arrays[0][i], "features_2": arrays[1][i], "features_3": arrays[2][i], "features_4": arrays[3][i]}
                drift_results.append(drift_result)
            self.reinstall()
            drift_results_iterator = iter(drift_results)                        

           # run axai
            if explain:
                explanation_results = []
                for img, shape in zip(imgs, shapes):
                    exp_mask = self.AXAI.explain(img,(shape[1], shape[0]))
                    explanation = {"maskRLE": [exp_mask], "dimensions": {"height": shape[0], "width": shape[1]}}
                    explanation_results.append(explanation)
                explanation_results_iterator = iter(explanation_results)
        
        # compile inference predictions, explanations, and drift output into a single output
        outputs = []    
        for j in range(len(model_inputs)):
            if j in indexed_errors:
                outputs.append(indexed_errors[j])
            else:
                inference_result = next(formatted_results_iterator)
                drift_result = next(drift_results_iterator)
                if explain:
                    explanation_result = next(explanation_results_iterator)
                else:
                    explanation_result = None
                output_item = get_success_json_structure(inference_result, explanation_result, drift_result)
                outputs.append(output_item)
        
        return outputs

    