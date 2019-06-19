import unittest
import onnxpipeline
import config
import os.path as osp
import os
import json

class notebook_test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_constructor_pass(self):
        directory_name = 'pytorch'
        pipeline = onnxpipeline.Pipeline(directory_name, print_logs=False)
        self.assertEquals(osp.join(os.getcwd(), directory_name), pipeline.path)

    def test_constructor_fail(self):
        directory_name = 'not_exist_directory'
        try:
            pipeline = onnxpipeline.Pipeline(directory_name)
            self.fail("Pipeline should raised RuntimeError")
        except:
            self.assertRaises(RuntimeError)

    def check_converted_json(self, output_json):
        with open(output_json) as f:
            data = json.load(f)
            conversion_status = data['conversion_status']
            correctness_verified = data['correctness_verified']
        return conversion_status, correctness_verified
    def check_json_staus(self, expected_status, json_status):
        conversion_status, correctness_verified = json_status
        self.assertEquals(expected_status[0], conversion_status)
        self.assertEquals(expected_status[1], correctness_verified)

    """
    def get_perf_test(self, pipeline, model):
        result = pipeline.perf_test(model=model, result="output.txt")
        return result
    """
    def test_convert_from_onnx(self):
        pipeline = onnxpipeline.Pipeline('onnx')
        def test_convert_pass():
            model = pipeline.convert_model(model_type='pytorch', model='model.onnx', model_input_shapes='(1,3,224,224)')
            return model

        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'UNSUPPORTED'], self.check_converted_json(output_json))


    def test_pytorch_pass(self):
        pipeline = onnxpipeline.Pipeline('pytorch', print_logs=False)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='pytorch', model='saved_model.pb', model_input_shapes='(1,3,224,224)')
            return model

        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))
        #result = self.test_perf_test(pipeline, model)

    def test_pytorch_fail(self):
        pipeline = onnxpipeline.Pipeline('pytorch', convert_directory='test_fail')
        def test_convert_fail_no_shapes():
            model = pipeline.convert_model(model_type='pytorch', model='saved_model.pb') #model_input_shapes='(1,3,224,224)')
            return model
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='pytorch', model_input_shapes='(1,3,224,224)')
            return model    
        
        model = test_convert_fail_no_shapes()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

    def test_tensorflow_pass(self):
        pipeline = onnxpipeline.Pipeline('mnist/model', print_logs=False)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='tensorflow')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_tensorflow_fail(self):
        pipeline = onnxpipeline.Pipeline('mnist/model', convert_directory='test_fail')
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='tensorflow', model='not_exist_path')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

    def test_cntk_pass(self):
        pipeline = onnxpipeline.Pipeline('cntk', print_logs=False)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='cntk', model='ResNet50_ImageNet_Caffe.model')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_cntk_fail(self):
        pipeline = onnxpipeline.Pipeline('cntk', convert_directory='test_fail', print_logs=False)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='cntk')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))