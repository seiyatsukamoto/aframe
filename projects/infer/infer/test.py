from data import Sequence

background_fname = '/home/seiya.tsukamoto/aframe/runs/bbh/data/test/background/background-1241443783-10796.hdf5'
injection_set_fname = '/home/seiya.tsukamoto/aframe/runs/bbh/data/test/waveforms.hdf5'
ifos = ['H1', 'L1']
shifts = [0, 1]
inference_sampling_rate = 4
batch_size = 128
rate = 70
output_shapes={'y': (), 'heatmap': (192,)}

seq = Sequence(background_fname = background_fname,
               injection_set_fname = injection_set_fname,
               ifos = ifos,
               shifts = shifts,
               inference_sampling_rate = inference_sampling_rate,
               batch_size = batch_size,
               rate = rate,
               output_shapes = output_shapes)