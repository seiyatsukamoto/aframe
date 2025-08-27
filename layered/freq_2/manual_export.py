import io
import logging
from typing import Optional

import h5py
import hermes.quiver as qv
import torch

from export.snapshotter import add_streaming_input_preprocessor
from utils.s3 import open_file

def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)

fduration=1.0
kernel_length=1.0
inference_sampling_rate=4.0
sample_rate=2048.0
batch_file=None
streams_per_gpu=6
aframe_instances=6
preproc_instances=1
clean=False ###
batch_size = 128
psd_length=64.0
highpass=32.0
lowpass=1024.0
q = None
fftlength = None
ifos=["H1", "L1"]
repository_directory='/home/seiya.tsukamoto/aframe/layered/freq_2/results/model_repo'
train_task='Train'
platform='TENSORRT'
weights = '/home/seiya.tsukamoto/aframe/layered/freq_2/training/model.pt'

with open_file(weights, "rb") as f:
    graph = nn = torch.jit.load(f, map_location="cpu")

repo = qv.ModelRepository('/home/seiya.tsukamoto/aframe/layered/freq_2/results/model_repo/', clean)
try:
    aframe = repo.models["aframe"]
except KeyError:
    aframe = repo.add("aframe", platform=platform)

if aframe_instances is not None:
    scale_model(aframe, aframe_instances)

batch_file = '/home/seiya.tsukamoto/aframe/layered/freq_2/training/batch.h5'
with open_file(batch_file, "rb") as f:
    batch_file = h5py.File(io.BytesIO(f.read()))

input_shape = batch_file['X'].shape
input_shape = (batch_size, input_shape[1], input_shape[2])

kwargs = {}
if platform == qv.Platform.ONNX:
    kwargs["opset_version"] = 13
    aframe.config.optimization.graph.level = -1
elif platform == qv.Platform.TENSORRT:
    kwargs["use_fp16"] = False

aframe.export_version(
    graph,
    input_shapes={"whitened": input_shape},
    output_names=["discriminator"],
    **kwargs,
)

ensemble_name = "aframe-stream"
try:
    ensemble = repo.models[ensemble_name]
except KeyError:
    ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
    fftlength = fftlength or kernel_length + fduration
    whitened = add_streaming_input_preprocessor(
        ensemble,
        aframe.inputs["whitened"],
        psd_length=psd_length,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        q=q,
        highpass=highpass,
        lowpass=lowpass,
        preproc_instances=preproc_instances,
        streams_per_gpu=streams_per_gpu,
    )
    ensemble.pipe(whitened, aframe.inputs["whitened"])
    ensemble.add_output(aframe.outputs["discriminator"])
    ensemble.export_version(None)
else:
    if aframe not in ensemble.models:
        raise ValueError(
            "Ensemble model '{}' already in repository "
            "but doesn't include model 'aframe'".format(ensemble_name)
        )
            
            
