import sys
sys.path.append('../')
import unittest
import onnxpipeline
import config
import os.path as osp
import os
import json
import shutil
import time

class notebook_test(unittest.TestCase):
    def setUp(self):
        self.convert_dir_pass = 'test_success'
        self.convert_dir_fail = 'test_fail'
        self.deep_dir = {
            'pytorch': 'pytorch',
            'tensorflow': 'mnist/model',
            'cntk': 'cntk', 
            'onnx': 'onnx',
            'caffe': 'caffe',
            'mxnet': 'mxnet',
            'sklearn': 'sklearn',
            'keras': 'KerasToONNX'
        }
        self.print_logs = False
        self.time_out = 30
    def tearDown(self):
        # remove created test dirtectories
        def remove_all_subfiles(directory_path):
            if not os.path.exists(directory_path):
                return 
            for path, subdir, files in os.walk(directory_path):
                for f in files:
                    try:
                        os.remove(osp.join(path, f))
                    except:
                        print("Cannot remove {}.".format(osp.join(path, f)))
                for d in subdir:
                    remove_all_subfiles(osp.join(path, d))

        for deep in self.deep_dir:
            for test_dir in [self.convert_dir_pass, self.convert_dir_fail]:
                directory_path = osp.join(os.getcwd(), self.deep_dir[deep], test_dir)
                remove_all_subfiles(directory_path)
        
        
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

    def wait_output_timeout(self, output_name):
        start = time.time()
        while not os.path.exists(output_name) and time.time() - start < self.time_out:
            time.sleep(0.3)
        if time.time() - start >= self.time_out:
            return False
        return True
    def check_converted_json(self, output_json):        
        if not self.wait_output_timeout(output_json):
            raise Exception('Fail. Convert model over {} seconds...'.format(self.time_out))

        with open(output_json) as f:
            data = json.load(f)
            conversion_status = data['conversion_status']
            correctness_verified = data['correctness_verified']
        return conversion_status, correctness_verified
    def check_json_staus(self, expected_status, json_status):
        conversion_status, correctness_verified = json_status
        self.assertEquals(expected_status[0], conversion_status)
        self.assertEquals(expected_status[1], correctness_verified)

    def check_latency_error(self, latency_name):
        if not self.wait_output_timeout(latency_name):
            raise Exception('Fail. Perf test over {} seconds...'.format(self.time_out))        
        with open(latency_name) as f:
            data = f.read()
            return data.count('error')
        
    
    def get_perftest(self, pipeline, model):
        result = pipeline.perf_test(model=model, result="result")
        return result
    
    def test_perf_test(self):
        directory_name = self.deep_dir['onnx']
        pipeline = onnxpipeline.Pipeline(directory_name)
        # generate input
        model = pipeline.convert_model(model_type='onnx', model='perf-test/ImageQnA.onnx', output_onnx_path='perf-test/ImageQnA.onnx')
        result_dir = pipeline.perf_test(model=model)
        latency_path = osp.join(result_dir, config.LATENCIES_TXT)
        self.assertEqual(self.check_latency_error(latency_path), 0) # 1 for cuda error, should be zerr


    def test_convert_from_onnx(self):
        directory_name = self.deep_dir['onnx']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='onnx', model='model.onnx')
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
        #result = self.test_perf_test(pipeline, model)

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

    def test_caffe_pass(self):
        directory_name = self.deep_dir['caffe']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='caffe', model='bvlc_alexnet.caffemodel', caffe_model_prototxt ='deploy.prototxt')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'UNSUPPORTED'], self.check_converted_json(output_json))

    def test_caffe_fail(self):
        directory_name = self.deep_dir['caffe']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail, print_logs=self.print_logs)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='caffe', caffe_model_prototxt ='deploy.prototxt')
            return model
        def test_convert_fail_no_prototxt():
            model = pipeline.convert_model(model_type='caffe', model='bvlc_alexnet.caffemodel')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

        model = test_convert_fail_no_prototxt()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))
    
    
    def test_sklearn_pass(self):
        directory_name = self.deep_dir['sklearn']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='scikit-learn', model='sklearn_svc.joblib', initial_types=("float_input", "FloatTensorType([1,4])"))
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_sklearn_fail(self):
        directory_name = self.deep_dir['sklearn']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail, print_logs=self.print_logs)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='scikit-learn', model='sklearn_svc.joblib')
            return model
        def test_convert_fail_no_type():
            model = pipeline.convert_model(model_type='scikit-learn', initial_types=("float_input", "FloatTensorType([1,4])"))
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))    

        model = test_convert_fail_no_type()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json)) 

    def test_keras_pass(self):
        directory_name = self.deep_dir['keras']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='keras', model='keras_Average_ImageNet.keras')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def test_keras_fail(self):
        directory_name = self.deep_dir['keras']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail, print_logs=self.print_logs)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='keras')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))
    
    # mxnet fails right now
    def futuretest_mxnet_pass(self):
        directory_name = self.deep_dir['mxnet']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_pass, print_logs=self.print_logs)
        def test_convert_pass():
            model = pipeline.convert_model(model_type='mxnet', model='resnet.json', model_params='resnet.params', model_input_shapes='(1,3,224,224)')
            return model
        model = test_convert_pass()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['SUCCESS', 'SUCCESS'], self.check_converted_json(output_json))

    def futuretest_mxnet_fail(self):
        directory_name = self.deep_dir['mxnet']
        pipeline = onnxpipeline.Pipeline(directory_name, convert_directory=self.convert_dir_fail, print_logs=self.print_logs)
        def test_convert_fail_no_model():
            model = pipeline.convert_model(model_type='mxnet', model_params='resnet.params', model_input_shapes='(1,3,224,224)')
            return model
        def test_convert_fail_no_shapes():
            model = pipeline.convert_model(model_type='mxnet',  model='resnet.json', model_params='resnet.params')
            return model

        def test_convert_fail_no_params():
            model = pipeline.convert_model(model_type='mxnet', model='resnet.json', model_input_shapes='(1,3,224,224)')
            return model

        model = test_convert_fail_no_model()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

        model = test_convert_fail_no_shapes()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))

        model = test_convert_fail_no_params()
        output_json = osp.join(pipeline.path, pipeline.convert_directory, config.OUTPUT_JSON)
        self.check_json_staus(['FAILED', 'FAILED'], self.check_converted_json(output_json))        
    