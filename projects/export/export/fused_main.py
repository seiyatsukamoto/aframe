import io
import logging
import h5py
import torch
from utils.s3 import open_file
from typing import TYPE_CHECKING, Optional

import hermes.quiver as qv
from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils

from utils.preprocessing import BackgroundSnapshotter

if TYPE_CHECKING:
    from hermes.quiver.model import EnsembleModel, ExposedTensor

def scale_model(model, instances):
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def export(
    weights: str,
    repository_directory: str,
    batch_file: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    preprocessor: torch.nn.Module,
    streams_per_gpu: int = 1,
    num_outputs: Optional[int] = 1,
    aframe_instances: Optional[int] = None,
    preproc_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.TENSORRT,
    clean: bool = False,
    verbose: bool = False,
    **kwargs,
) -> None:
    logging.info("Initializing model graph")
    with open_file(weights, "rb") as f:
        graph = nn = torch.jit.load(f, map_location="cpu")
    
    graph.eval()
    logging.info(f"Initialize:\n{nn}")
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        aframe = repo.models["aframe"]
    except KeyError:
        aframe = repo.add("aframe", platform=platform)
    
    if aframe_instances is not None:
        scale_model(aframe, aframe_instances)
    
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13
        aframe.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = False
    
    if num_outputs < 1:
        raise ValueError("num_outputs must be >= 1")
    
    output_names = (
        ["discriminator"]
        if num_outputs == 1
        else [f"discriminator_{i}" for i in range(num_outputs)]
    )
    
    input_shape = {'strain': (2, num_ifos, int(2048*(psd_length+(batch_size-1)/inference_sampling_rate+kernel_length+fduration)))}
    aframe.export_version(
        graph,
        input_shapes=input_shape_dict,
        output_names=output_names,
        **kwargs,
    )
    
    ensemble_name = "aframe-stream"
    try:
        ensemble = repo.models[ensemble_name]
    except KeyError:
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
        snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        )
        stride = int(sample_rate / inference_sampling_rate)
        state_shape = (2, num_ifos, snapshotter.state_size)
        streaming_model = streaming_utils.add_streaming_model(
            ensemble.repository,
            streaming_layer=snapshotter,
            name="snapshotter",
            input_name="stream",
            input_shape=input_shape,
            state_names=["snapshot"],
            state_shapes=[state_shape],
            output_names=["strain"],
            streams_per_gpu=streams_per_gpu,
        )
        ensemble.add_input(streaming_model.inputs["stream"])
        ensemble.pipe(
            streaming_model.outputs["strain"],
            aframe.inputs["strain"],
        )
        for name in output_names:
            ensemble.add_output(aframe.outputs[name])
        
        ensemble.export_version(None)
    else:
        if aframe not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
    
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e+8
    )
    snapshotter.config.parameters["intra_op_thread_count"].string_value = "1"
    snapshotter.config.write()