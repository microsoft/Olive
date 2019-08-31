# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
sys.path.append('../../utils')
import unittest
import onnxpipeline
import config
import os.path as osp
import os
import json
import shutil

class pipeline_test(unittest.TestCase):
    def setUp(self):
        self.convert_dir_pass = 'test_success'
        self.convert_dir_fail = 'test_fail'
        self.deep_dir = {
            'pytorch': 'pytorch',
            'tensorflow': 'mnist/model',
            'cntk': 'cntk', 
            'onnx': 'onnx'
        }
        self.print_logs = False
    def tearDown(self):
        # remove created test dirtectories
        def remove_all_subfiles(directory_path):
            shutil.rmtree(directory_path)
        for deep in self.deep_dir:
            for test_dir in [self.convert_dir_pass, self.convert_dir_fail]:
                directory_path = osp.join(os.getcwd(), self.deep_dir[deep], test_dir)
                try:
                    remove_all_subfiles(directory_path)
                except:
                    print("Cannot remove {}.".format(directory_path))
        
        
    def test_constructor_pass(self):
        directory_name = self.deep_dir['pytorch']
        pipeline = onnxpipeline.Pipeline(directory_name, print_logs=self.print_logs)
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
    def get_perf_tuning(self, pipeline, model):
        result = pipeline.perf_tuning(model=model, result="output.txt")
        return result
    """
    def test_convert_from_onnx(self):
        directory_name = self.deep_dir['onnx']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='onnx', model='model.onnx', model_input_shapes='(1,3,224,224)')
            return model

        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SKIPPED'], self.check_converted_json(output_json))


    def test_pytorch_pass(self):
        directory_name = self.deep_dir['pytorch']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='pytorch', model='saved_model.pb', model_input_shapes='(1,3,224,224)')
            return model

        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))
        #result = self.test_perf_tuning(pipeline, model)

    def test_pytorch_fail(self):
        directory_name = self.deep_dir['pytorch']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail)
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
        directory_name = self.deep_dir['tensorflow']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='tensorflow')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_tensorflow_fail(self):
        directory_name = self.deep_dir['tensorflow']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='tensorflow', model='not_exist_path')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

    def test_cntk_pass(self):
        directory_name = self.deep_dir['cntk']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='cntk', model='ResNet50_ImageNet_Caffe.model')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_cntk_fail(self):
        directory_name = self.deep_dir['cntk']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail, print_logs=self.print_logs)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='cntk')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))