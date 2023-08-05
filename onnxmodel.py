import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from scipy.io import wavfile
from rfftr import rfft_function

def create_sample_model(sample_rate, audio_data):

    if audio_data.dtype != np.float32:
      audio_data = audio_data.astype(np.float32)

    input_shape = audio_data.shape
    input_tensor = numpy_helper.from_array(audio_data, name='input_tensor')
    ip = input_shape[0]

    rfft_result, magnitude_shape = rfft_function(audio_data)
    mg = magnitude_shape[0]

    rfft_result_tensor = numpy_helper.from_array(rfft_result, name='rfft_result_tensor')
    magnitude_shape_tensor = numpy_helper.from_array(np.array(rfft_result,dtype=np.float32), name='magnitude_shape_tensor')

    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 10

    graph_def = helper.make_graph(
        [],
        'sample_model',
        inputs=[helper.make_tensor_value_info('input_tensor', onnx.TensorProto.FLOAT, [ip])],
        outputs=[
            helper.make_tensor_value_info('magnitude_shape_tensor', onnx.TensorProto.FLOAT, [mg])],
        initializer=[input_tensor,magnitude_shape_tensor],
    )

    model_def = helper.make_model(graph_def, producer_name='sample_model')

    onnx.save_model(model_def, 'rfft_model.onnx')

if __name__ == '__main__':
    sample_rate, audio_data = wavfile.read('audio1.wav')
    create_sample_model(sample_rate, audio_data)
