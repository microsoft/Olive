import unittest
import onnxpipeline
import config
import os.path as osp
import os
import json

class notebook_test(unittest.TestCase):
    def setUp(self):
        #self.args = (3, 2)
        pass
    def tearDown(self):
        #self.args = None
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

    def test_pytorch_pass(self):
        pipeline = onnxpipeline.Pipeline('pytorch', print_logs=False)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='pytorch', model='saved_model.pb', model_input_shapes='(1,3,224,224)')
            return model
        def test_perf_test(model):
            result = pipeline.perf_test(model=model, result="output.txt")
            return result
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        conversion_status, correctness_verified = self.check_converted_json(output_json)
        self.assertEquals('SUCCESS', conversion_status)
        self.assertEquals('SUCCESS', correctness_verified)
        #result = test_perf_test(model)

    def test_pytorch_fail(self):
        pipeline = onnxpipeline.Pipeline('pytorch', convert_directory='test_fail')
        def test_convert_pass():
            model = pipeline.convert_model(model_type='pytorch', model='saved_model.pb') #model_input_shapes='(1,3,224,224)')
            return model
        def test_perf_test(model):
            result = pipeline.perf_test(model=model, result="output.txt")
            return result
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        print(output_json)
        conversion_status, correctness_verified = self.check_converted_json(output_json)
        self.assertEquals('FAILED', conversion_status)
        self.assertEquals('FAILED', correctness_verified)
        