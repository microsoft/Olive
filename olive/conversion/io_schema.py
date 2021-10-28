class IOSchema:

    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape

class IOSchemaLoader:
    NAME_KEY = "name"
    DTYPE_KEY = "dataType"
    SHAPE_KEY = "shape"

    def __init__(self, inputs_schema, outputs_schema):
        self.input_schema = inputs_schema
        self.output_schema = outputs_schema

    @staticmethod
    def validate_schema_properties(schema, properties):
        if not schema or len(schema) == 0:
            return False

        for i in schema:
            if (not i.name) and (IOSchemaLoader.NAME_KEY in properties):
                return False
            if (not i.dtype) and (IOSchemaLoader.DTYPE_KEY in properties):
                return False
            if (not i.shape) and (IOSchemaLoader.SHAPE_KEY in properties):
                return False

        return True
