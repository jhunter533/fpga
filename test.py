import hls4ml
config=hls4ml.utils.config_from_pytorch_model('../Project-Wintermute/physics simulator/gymnasium/SAC/SAC_Actor_0.pth',input_shape=([24],[256],[296]),backend='Vitis')
hls_model=hls4ml.converters.pytorch_to_hls(config)
hls_model.build()
hls4ml.report.read_vivado_report('my-hls-test')
