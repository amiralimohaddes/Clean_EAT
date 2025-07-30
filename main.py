
import os
from load_dataset import load_data
import torch
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from aed_sklearn_transformer_forest.features import time, spectral, general
# Import your custom modules for nearest_neighbor
# from your_module import nearest_neighbor


# Placeholder for EAT PyTorch model loader (replace with your actual model class if needed)
class EATModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

def get_deep_feat_cls_eat_all_model(feat_model_type, feat_model_depth, n_components=2, model_type='kmeansNN'):
    # Use .pt EAT model instead of ONNX
    eat_model_path = f'pt_files/EAT/DeepFeats_{feat_model_type}_{feat_model_depth}.pt'
    eat_model = EATModel(eat_model_path)

    # Always use kmeansNN for anomaly detection
    model = nearest_neighbor.KMeansNN(n_clusters=n_components)

    pipe = Pipeline([
        ('resample', time.PolyphaseResample(up=160,down=441,filter_len=160,window=("kaiser", 12))),
        ('frame_audio', general.FrameDimension(frame_size=400,hop_size=160,axis=1)),
        ('reduce_dims', FeatureUnion([
            ('mel_pipe', Pipeline([
                ('reduce', general.ReduceAxes(axis_to_keep=2)),
                ('frame', time.Window(n_features=400,window_type='hann')),
                ('dft', time.DFT(nfft=400,)),
                ('magnitude', spectral.Magnitude()),
                ('mel', spectral.MEL(nfft=400, nmels=128, samplerate=16000)),
                ('norm', general.LinearScaler(offset=-4.288,scale=4.469)),
            ])),
            ('shape_info', general.ShapeInfoExtractor(axis_to_keep=2)),
        ])),
        ('expand', general.ExpandAxes(feature_axis=2,batch_expand_dims=2)),
        ('deep_feats', eat_model),  # Use PyTorch EAT model
        ('feature_mask', general.DimensionalSlicing(start=0,end=1,step=1,axis=1)),
        ('frame_unbuffer', general.ReduceAxes(axis_to_keep=-1)),
        ('scaler', StandardScaler()),
        ('AnodetecModel', model),
        ('out_scale', MinMaxScaler()),
    ])
    custome_transfromer_list = [
        time.PolyphaseResample,
        time.Window,
        time.DFT,
        spectral.Magnitude,
        spectral.MEL,
        general.LinearScaler,
        general.FrameDimension,
        general.DimensionalSlicing,
        EATModel,
        general.ReduceAxes,
        general.ExpandAxes,
        general.ShapeInfoExtractor,
        nearest_neighbor.KMeansNN,
        # Add other custom transformers as needed
    ]
    return pipe, custome_transfromer_list


def prepare_X(audio_list):
    # This function should convert a list of audio arrays to the expected input for the pipeline
    # For now, just stack as numpy arrays (adjust as needed for your pipeline)
    return [np.array(a) for a in audio_list if a is not None]

if __name__ == "__main__":
    dataset_root = "./dataset"
    data = load_data(dataset_root)
    pipe, _ = get_deep_feat_cls_eat_all_model('audioset', 'base', n_components=2)

    # Example: process all train data for 'fan'
    train_audio = [audio for path, audio in data['fan']['train'] if audio is not None]
    test_audio = [audio for path, audio in data['fan']['test'] if audio is not None]

    X_train = prepare_X(train_audio)
    X_test = prepare_X(test_audio)

    # Fit pipeline on train data
    if len(X_train) > 0:
        pipe.fit(X_train)
        print("Pipeline fitted on train data.")
    else:
        print("No train data available.")

    # Predict or transform test data
    if len(X_test) > 0:
        results = pipe.transform(X_test)
        print("Test results:", results)
    else:
        print("No test data available.")
